"""Pydantic schemas used by agents that produce structured output.

The framework's primary artifact is still prose: each agent's natural-language
reasoning is what users read in the saved markdown reports and what the
downstream agents read as context.  Structured output is layered onto the
three decision-making agents (Research Manager, Trader, Portfolio Manager)
so that:

- Their outputs follow consistent section headers across runs and providers
- Each provider's native structured-output mode is used (json_schema for
  OpenAI/xAI, response_schema for Gemini, tool-use for Anthropic)
- Schema field descriptions become the model's output instructions, freeing
  the prompt body to focus on context and the rating-scale guidance
- A render helper turns the parsed Pydantic instance back into the same
  markdown shape the rest of the system already consumes, so display,
  memory log, and saved reports keep working unchanged
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from tradingagents.agents.utils.evidence_policy import UNVERIFIED_MARKER

# LLMs sometimes write a placeholder string ("None", "N/A", ...) into an optional
# numeric field instead of omitting it. Coerce those to None so the structured
# call validates instead of erroring (#1058). Pydantic still parses real numeric
# strings ("189.5") to float.
_NULLISH_FLOAT = {"", "none", "n/a", "na", "null", "nil", "-", "tbd", "unknown"}


def _coerce_optional_float(value):
    if isinstance(value, str) and value.strip().lower() in _NULLISH_FLOAT:
        return None
    return value


# ---------------------------------------------------------------------------
# Shared rating types
# ---------------------------------------------------------------------------


class PortfolioRating(str, Enum):
    """5-tier rating used by the Research Manager and Portfolio Manager."""

    BUY = "Buy"
    OVERWEIGHT = "Overweight"
    HOLD = "Hold"
    UNDERWEIGHT = "Underweight"
    SELL = "Sell"


class TraderAction(str, Enum):
    """3-tier transaction direction used by the Trader.

    The Trader's job is to translate the Research Manager's investment plan
    into a concrete transaction proposal: should the desk execute a Buy, a
    Sell, or sit on Hold this round.  Position sizing and the nuanced
    Overweight / Underweight calls happen later at the Portfolio Manager.
    """

    BUY = "Buy"
    HOLD = "Hold"
    SELL = "Sell"


class PositionIntent(str, Enum):
    """What a Buy/Sell actually does to the book.

    ``Sell`` is ambiguous on its own: it can mean opening a short or scaling
    out of an existing long, and those two have opposite stop-loss geometry.
    A shipped proposal mixed them — action Sell, entry 520, stop 540, sizing
    "scale out of the existing long" — which is a short's stop attached to a
    long's exit. Naming the intent makes the contradiction checkable.
    """

    OPEN_LONG = "Open or add to a long"
    REDUCE_LONG = "Reduce or exit an existing long"
    OPEN_SHORT = "Open or add to a short"
    REDUCE_SHORT = "Reduce or cover an existing short"
    NO_CHANGE = "No position change"


# Which intents each action can legitimately express.
_ACTION_INTENTS: dict[TraderAction, frozenset[PositionIntent]] = {
    TraderAction.BUY: frozenset({PositionIntent.OPEN_LONG, PositionIntent.REDUCE_SHORT}),
    TraderAction.SELL: frozenset({PositionIntent.REDUCE_LONG, PositionIntent.OPEN_SHORT}),
    TraderAction.HOLD: frozenset({PositionIntent.NO_CHANGE}),
}

# Which side of the entry a protective stop must sit on. A stop protecting long
# exposure is below the entry; a stop protecting short exposure is above it.
# ``REDUCE_*`` keeps the residual position's geometry: trimming a long still
# leaves a long to protect.
_STOP_BELOW_ENTRY: dict[PositionIntent, bool] = {
    PositionIntent.OPEN_LONG: True,
    PositionIntent.REDUCE_LONG: True,
    PositionIntent.OPEN_SHORT: False,
    PositionIntent.REDUCE_SHORT: False,
}


# ---------------------------------------------------------------------------
# Research Manager
# ---------------------------------------------------------------------------


class ResearchPlan(BaseModel):
    """Structured investment plan produced by the Research Manager.

    Hand-off to the Trader: the recommendation pins the directional view,
    the rationale captures which side of the bull/bear debate carried the
    argument, and the strategic actions translate that into concrete
    instructions the trader can execute against.
    """

    recommendation: PortfolioRating = Field(
        description=(
            "The investment recommendation. Exactly one of Buy / Overweight / "
            "Hold / Underweight / Sell. Reserve Hold for situations where the "
            "evidence on both sides is genuinely balanced; otherwise commit to "
            "the side with the stronger arguments."
        ),
    )
    rationale: str = Field(
        description=(
            "Conversational summary of the key points from both sides of the "
            "debate, ending with which arguments led to the recommendation. "
            "Speak naturally, as if to a teammate."
        ),
    )
    strategic_actions: str = Field(
        description=(
            "Concrete steps for the trader to implement the recommendation, "
            "including position sizing guidance consistent with the rating."
        ),
    )


def render_research_plan(plan: ResearchPlan) -> str:
    """Render a ResearchPlan to markdown for storage and the trader's prompt context."""
    return "\n".join([
        f"**Recommendation**: {plan.recommendation.value}",
        "",
        f"**Rationale**: {plan.rationale}",
        "",
        f"**Strategic Actions**: {plan.strategic_actions}",
    ])


# ---------------------------------------------------------------------------
# Trader
# ---------------------------------------------------------------------------


class TraderProposal(BaseModel):
    """Structured transaction proposal produced by the Trader.

    The trader reads the Research Manager's investment plan and the analyst
    reports, then turns them into a concrete transaction: what action to
    take, the reasoning that justifies it, and the practical levels for
    entry, stop-loss, and sizing.
    """

    action: TraderAction = Field(
        description="The transaction direction. Exactly one of Buy / Hold / Sell.",
    )
    position_intent: PositionIntent = Field(
        description=(
            "What this action does to the book. Buy must be 'Open or add to a "
            "long' or 'Reduce or cover an existing short'; Sell must be 'Reduce "
            "or exit an existing long' or 'Open or add to a short'; Hold must be "
            "'No position change'. Choose deliberately: a Sell that trims an "
            "existing long and a Sell that opens a new short need stop-loss "
            "levels on opposite sides of the entry price."
        ),
    )
    reasoning: str = Field(
        description=(
            "The case for this action, anchored in the analysts' reports and "
            "the research plan. Two to four sentences."
        ),
    )
    entry_price: float | None = Field(
        default=None,
        description=(
            "Optional entry price target in the instrument's quote currency. "
            "Must be consistent with the verified price levels in the prompt."
        ),
    )
    stop_loss: float | None = Field(
        default=None,
        description=(
            "Optional stop-loss price in the instrument's quote currency. It must "
            "sit BELOW entry_price when the intent is long-side (open or reduce a "
            "long) and ABOVE entry_price when the intent is short-side. Size the "
            "distance against the ATR given in the prompt — a stop closer than "
            "1x ATR is inside normal daily noise and will usually be triggered."
        ),
    )
    position_sizing: str | None = Field(
        default=None,
        description=(
            "Optional sizing guidance, e.g. '5% of portfolio'. Describe the same "
            "operation named in position_intent — do not describe trimming a long "
            "here while the intent says opening a short."
        ),
    )

    @field_validator("entry_price", "stop_loss", mode="before")
    @classmethod
    def _nullish_float_to_none(cls, v):
        return _coerce_optional_float(v)

    @model_validator(mode="after")
    def _check_internal_consistency(self):
        """Reject proposals whose fields contradict each other.

        Raising here makes the provider's structured-output layer surface the
        error and retry, which is the right moment to fix a self-contradictory
        trade — far better than shipping it and catching the contradiction in a
        reader's head three reports later.
        """
        allowed = _ACTION_INTENTS.get(self.action, frozenset())
        if self.position_intent not in allowed:
            options = " or ".join(sorted(intent.value for intent in allowed))
            raise ValueError(
                f"action '{self.action.value}' cannot have position_intent "
                f"'{self.position_intent.value}'. Use: {options}."
            )

        if self.entry_price is not None and self.stop_loss is not None:
            stop_below = _STOP_BELOW_ENTRY.get(self.position_intent)
            if stop_below is True and self.stop_loss >= self.entry_price:
                raise ValueError(
                    f"position_intent '{self.position_intent.value}' protects long "
                    f"exposure, so stop_loss ({self.stop_loss}) must be BELOW "
                    f"entry_price ({self.entry_price})."
                )
            if stop_below is False and self.stop_loss <= self.entry_price:
                raise ValueError(
                    f"position_intent '{self.position_intent.value}' protects short "
                    f"exposure, so stop_loss ({self.stop_loss}) must be ABOVE "
                    f"entry_price ({self.entry_price})."
                )
        return self


# Below this many ATRs, a stop sits inside the instrument's ordinary daily
# range and will usually be taken out by noise rather than by the thesis
# failing.
MIN_STOP_ATR_MULTIPLE = 1.0


def render_risk_check(proposal: TraderProposal, levels) -> list[str]:
    """Verify the proposal's levels against measured volatility, in Python.

    The model states an ATR multiple in prose; this computes it. A shipped
    report claimed a stop was "1-1.5x ATR" when the distance was 0.5x — the
    kind of claim that reads as disciplined risk management while describing a
    stop that daily noise would take out. ``levels`` is a
    ``dataflows.market_data_validator.TradeReference`` or None.
    """
    if levels is None or proposal.entry_price is None or proposal.stop_loss is None:
        return []

    distance = abs(proposal.entry_price - proposal.stop_loss)
    multiple = levels.atr_multiple(proposal.entry_price, proposal.stop_loss)

    lines = [
        "",
        "**Risk Check** (computed from verified market data, not model-stated):",
        f"- Reference close ({levels.as_of}, bar status {levels.bar_status}): "
        + (f"{levels.close:,.2f}" if levels.close is not None else "N/A"),
        "- ATR: " + (f"{levels.atr:,.2f}" if levels.atr else "N/A"),
        f"- Stop distance: |{proposal.entry_price:,.2f} - {proposal.stop_loss:,.2f}| "
        f"= {distance:,.2f}"
        + (f" = {multiple:.2f}x ATR" if multiple is not None else ""),
    ]
    if multiple is not None and multiple < MIN_STOP_ATR_MULTIPLE:
        lines.append(
            f"- ⚠️ The stop is {multiple:.2f}x ATR from entry, inside the instrument's "
            f"ordinary daily range. Normal noise is likely to trigger it before the "
            f"thesis is tested. A stop of at least {MIN_STOP_ATR_MULTIPLE:.1f}x ATR "
            f"would sit at "
            + (
                f"{proposal.entry_price - levels.atr:,.2f}"
                if _STOP_BELOW_ENTRY.get(proposal.position_intent, True)
                else f"{proposal.entry_price + levels.atr:,.2f}"
            )
            + "."
        )
    if levels.close is not None and proposal.entry_price is not None:
        drift = abs(proposal.entry_price - levels.close)
        if levels.atr and drift > 2 * levels.atr:
            lines.append(
                f"- ⚠️ The entry price is {drift:,.2f} ({drift / levels.atr:.1f}x ATR) "
                f"away from the last verified close of {levels.close:,.2f}."
            )
    return lines


def render_trader_proposal(proposal: TraderProposal, levels=None) -> str:
    """Render a TraderProposal to markdown, with a computed risk check.

    The trailing ``FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**`` line is
    preserved for backward compatibility with the analyst stop-signal text
    and any external code that greps for it.
    """
    parts = [
        f"**Action**: {proposal.action.value}",
        "",
        f"**Position Intent**: {proposal.position_intent.value}",
        "",
        f"**Reasoning**: {proposal.reasoning}",
    ]
    if proposal.entry_price is not None:
        parts.extend(["", f"**Entry Price**: {proposal.entry_price}"])
    if proposal.stop_loss is not None:
        parts.extend(["", f"**Stop Loss**: {proposal.stop_loss}"])
    if proposal.position_sizing:
        parts.extend(["", f"**Position Sizing**: {proposal.position_sizing}"])
    parts.extend(render_risk_check(proposal, levels))
    parts.extend([
        "",
        f"FINAL TRANSACTION PROPOSAL: **{proposal.action.value.upper()}**",
    ])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Portfolio Manager
# ---------------------------------------------------------------------------


class PortfolioDecision(BaseModel):
    """Structured output produced by the Portfolio Manager.

    The model fills every field as part of its primary LLM call; no separate
    extraction pass is required. Field descriptions double as the model's
    output instructions, so the prompt body only needs to convey context and
    the rating-scale guidance.
    """

    rating: PortfolioRating = Field(
        description=(
            "The final position rating. Exactly one of Buy / Overweight / Hold / "
            "Underweight / Sell, picked based on the analysts' debate."
        ),
    )
    executive_summary: str = Field(
        description=(
            "A concise action plan covering entry strategy, position sizing, "
            "key risk levels, and time horizon. Two to four sentences."
        ),
    )
    investment_thesis: str = Field(
        description=(
            "Detailed reasoning anchored in specific evidence from the analysts' "
            "debate. If prior lessons are referenced in the prompt context, "
            "incorporate them; otherwise rely solely on the current analysis."
        ),
    )
    price_target: float | None = Field(
        default=None,
        description="Optional target price in the instrument's quote currency.",
    )
    time_horizon: str | None = Field(
        default=None,
        description="Optional recommended holding period, e.g. '3-6 months'.",
    )

    @field_validator("price_target", mode="before")
    @classmethod
    def _nullish_float_to_none(cls, v):
        return _coerce_optional_float(v)


def render_pm_decision(decision: PortfolioDecision) -> str:
    """Render a PortfolioDecision back to the markdown shape the rest of the system expects.

    Memory log, CLI display, and saved report files all read this markdown,
    so the rendered output preserves the exact section headers (``**Rating**``,
    ``**Executive Summary**``, ``**Investment Thesis**``) that downstream
    parsers and the report writers already handle.
    """
    parts = [
        f"**Rating**: {decision.rating.value}",
        "",
        f"**Executive Summary**: {decision.executive_summary}",
        "",
        f"**Investment Thesis**: {decision.investment_thesis}",
    ]
    if decision.price_target is not None:
        parts.extend(["", f"**Price Target**: {decision.price_target}"])
    if decision.time_horizon:
        parts.extend(["", f"**Time Horizon**: {decision.time_horizon}"])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Sentiment Analyst
# ---------------------------------------------------------------------------


class SentimentBand(str, Enum):
    """Discrete sentiment direction produced by the Sentiment Analyst.

    Six tiers keep the signal granular enough to be actionable while remaining
    small enough for every provider to map reliably from its JSON output.
    """

    BULLISH = "Bullish"
    MILDLY_BULLISH = "Mildly Bullish"
    NEUTRAL = "Neutral"
    MIXED = "Mixed"
    MILDLY_BEARISH = "Mildly Bearish"
    BEARISH = "Bearish"


class SentimentReport(BaseModel):
    """Structured sentiment report produced by the Sentiment Analyst.

    Replaces the previous free-form prose output so downstream consumers
    (dashboards, audit logs, PDF renderers, other agents) can read
    ``overall_band`` and ``overall_score`` without maintaining fragile regex
    fallbacks that drift with every model release. ``narrative`` preserves the
    rich source-by-source analysis; ``render_sentiment_report`` prepends a
    deterministic header so the saved report stays human-readable.
    """

    overall_band: SentimentBand = Field(
        description=(
            "Overall sentiment direction. Exactly one of: "
            "Bullish / Mildly Bullish / Neutral / Mixed / Mildly Bearish / Bearish. "
            "Use Mixed when sources point in clearly different directions. "
            "Use Neutral only when all sources are genuinely silent or non-committal."
        ),
    )
    overall_score: float = Field(
        ge=0.0,
        le=10.0,
        description=(
            "Numeric sentiment intensity on a 0–10 scale. "
            "0 = maximally bearish, 5 = neutral, 10 = maximally bullish. "
            "Guideline for consistency with overall_band: "
            "Bullish ~6.5–10, Mildly Bullish ~5.5–6.4, Neutral/Mixed ~4.5–5.5, "
            "Mildly Bearish ~3.5–4.4, Bearish ~0–3.4. "
            "Only the 0–10 bounds are enforced."
        ),
    )
    confidence: Literal["low", "medium", "high"] = Field(
        description=(
            "Confidence in the assessment based on data quality and sample size. "
            "Use 'low' when one or more sources returned a placeholder or fewer "
            "than 5 data points; 'medium' when data is present but sparse; "
            "'high' when all three sources returned substantive data."
        ),
    )
    unverified_numeric_claims: list[str] = Field(
        default_factory=list,
        description=(
            "Every quantitative claim you relayed whose only source is a "
            "StockTwits or Reddit post — growth rates, margins, multiples, "
            "cash-flow figures, price targets. One short line each, stating the "
            "claim and the platform, e.g. 'free cash flow fell 39% — StockTwits "
            "poster'. Downstream agents use this list to avoid restating these "
            "numbers as fact. Empty list if the social sources made no "
            "quantitative claims."
        ),
    )
    narrative: str = Field(
        description=(
            "Full sentiment report covering, in order: "
            "(1) source-by-source breakdown with specific evidence (cite message "
            "counts, ratios, notable posts); "
            "(2) cross-source divergences and alignments; "
            "(3) dominant narrative themes; "
            "(4) catalysts and risks surfaced by the data; "
            "(5) a markdown table summarising key sentiment signals, their "
            "direction, source, and supporting evidence. "
            "Keep it informative and substantive: develop each section thoroughly "
            "with concrete evidence so every point adds new signal for the trader."
        ),
    )


def render_sentiment_report(report: SentimentReport) -> str:
    """Render a SentimentReport to the markdown shape the rest of the system expects.

    The structured header (band + score + confidence) is prepended to the
    narrative so the saved report is both human-readable and machine-parseable
    without regex.

    Unverified numeric claims are rendered as an explicit block rather than left
    inside the prose. Downstream agents read this report as context, and a
    number buried mid-paragraph gets lifted out and restated as fact — that is
    how a StockTwits poster's cash-flow figure became a pillar of a shipped
    Underweight rating.
    """
    parts = [
        f"**Overall Sentiment:** **{report.overall_band.value}** "
        f"(Score: {report.overall_score:.1f}/10)",
        f"**Confidence:** {report.confidence.capitalize()}",
    ]
    if report.unverified_numeric_claims:
        parts += [
            "",
            "**Unverified numeric claims from social sources** — these are what "
            "posters asserted, not measured facts. Do not restate them as fact or "
            "use them to support a recommendation; where they conflict with the "
            "verified fundamentals or market snapshot, the verified figure stands.",
        ]
        parts += [
            f"- {claim.strip()} {UNVERIFIED_MARKER}"
            if UNVERIFIED_MARKER not in claim
            else f"- {claim.strip()}"
            for claim in report.unverified_numeric_claims
        ]
    parts += ["", report.narrative]
    return "\n".join(parts)
