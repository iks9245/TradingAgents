"""Verified figures for the agents downstream of the analysts.

The analysts have had deterministic snapshots for a while: the market analyst
is handed verified OHLCV and indicators, the fundamentals analyst is handed
recomputed margins and ratios with their operands printed. Everything after
them read prose only.

That gap is what makes a wrong figure unrecoverable once it is written down.
The Bull and Bear researchers receive the same four report fields and neither
has a route back to the source, so a bad number arrives as a *shared premise*:
the bull argues from it, the bear argues against the conclusion rather than the
number, and the adversarial structure never touches it. Adversarial agents can
contest interpretations of a fact; they cannot contest the fact itself when
both sides were handed it and neither can look it up.

The Portfolio Manager is the sharpest case, because it sets a price target
while seeing only the debate. It was already given verified price levels after
inventing valuation arithmetic to justify one ("$450 implies about 40x 2025 PE"
when FY EPS made it 170x) — but price levels alone cannot check a P/E. The EPS
that would have caught it lives in the fundamentals snapshot, which nothing
downstream of the analysts could see.

This module resolves both snapshots once at run start, stores them on the
state, and hands them to the debate and decision agents with one rule attached:
where a report disagrees with the block, the block stands and the conflict gets
named. That gives at least one party in the debate something to check a premise
against, which is the only place in the pipeline where a premise can still be
challenged.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tradingagents.dataflows.fundamentals_validator import (
    render_fundamentals_snapshot_block,
)
from tradingagents.dataflows.market_data_validator import (
    get_trade_reference_levels,
    render_trade_reference_block,
)

# Kept separate from the block itself so the rule is identical for every
# consumer. The last sentence is the point of the whole mechanism: a debater
# who spots a bad premise is the only actor positioned to catch it, because
# every check after this point runs on prose that already contains it.
VERIFIED_EVIDENCE_RULE = """
**Using the verified figures.** The block(s) above were computed in Python from
the same price data and filings the analysts worked from, before any agent wrote
prose. They are not a summary of the reports — they are the source the reports
were supposed to draw from.

Where an analyst report, a debate turn, a news item, or a social post states a
figure that disagrees with a verified block, **the verified figure stands**. Name
the conflict explicitly — which source said what, and what the verified value is
— rather than averaging the two, silently preferring one, or reconciling them
into a third number. Quote a verified figure with the period or as-of date it
carries; a value labelled (TTM) or (MRQ) is not a fiscal-year figure.

Challenging a figure on this basis is not a digression from your argument. Every
check downstream of you runs on prose that already contains the error, so this is
the last point at which a wrong premise can be caught.
""".strip()

_MARKET_UNAVAILABLE = render_trade_reference_block(None, include_proposal_rule=False)

# Sourced from the fundamentals validator's own notice so the wording an agent
# sees is identical whether the lookup failed or was never made.
FUNDAMENTALS_UNAVAILABLE = render_fundamentals_snapshot_block("", "")


def resolve_verified_evidence(
    ticker: str, trade_date: str, asset_type: str = "stock"
) -> tuple[str, str]:
    """Resolve both verified blocks for a run. Returns ``(market, fundamentals)``.

    Called once per run from the entry point — ``propagate()`` or the CLI — not
    from inside the graph. Both underlying resolvers fail open, returning an
    explicit "unavailable" notice rather than raising, so a missing vendor
    response degrades the prompt instead of blocking the run. Crypto is the
    ordinary case for missing fundamentals, not an error.
    """
    if not ticker or not trade_date:
        return _MARKET_UNAVAILABLE, FUNDAMENTALS_UNAVAILABLE

    levels = get_trade_reference_levels(ticker, str(trade_date))
    market_block = render_trade_reference_block(levels, include_proposal_rule=False)

    # Crypto has no filings to verify against. Asking anyway returns the
    # unavailable notice, which is the right text to show — but skipping the
    # lookup avoids a pointless vendor round-trip on every crypto run.
    if asset_type == "crypto":
        return market_block, FUNDAMENTALS_UNAVAILABLE

    return market_block, render_fundamentals_snapshot_block(ticker, str(trade_date))


def get_verified_evidence_block(
    state: Mapping[str, Any], *, include_market: bool = True
) -> str:
    """Return the verified-evidence section for a downstream agent's prompt.

    Reads the blocks resolved once at run start and stored on the state (see
    ``TradingAgentsGraph.resolve_verified_evidence``). A state built without
    them — bare programmatic states, tests — degrades to the explicit
    "unavailable" notices rather than resolving here, following the same rule
    ``get_instrument_context_from_state`` states for the instrument context: a
    consumer must never be forced into a vendor lookup mid-graph. The
    degradation is visible in the prompt, so an agent told the figures are
    unavailable declines to state them instead of quietly inventing them.

    ``include_market=False`` is for the Trader and the Portfolio Manager, which
    already render their own price-levels block carrying the execution rule
    about ATR distances; repeating the levels would put the same numbers in one
    prompt twice under two different headings.
    """
    market = state.get("verified_market_block")
    fundamentals = state.get("verified_fundamentals_block")

    if not _is_present(market):
        market = _MARKET_UNAVAILABLE
    if not _is_present(fundamentals):
        fundamentals = FUNDAMENTALS_UNAVAILABLE

    sections = []
    if include_market:
        sections.append("<start_of_verified_market_data>\n" + market.strip() + "\n<end_of_verified_market_data>")
    sections.append(
        "<start_of_verified_fundamentals>\n"
        + fundamentals.strip()
        + "\n<end_of_verified_fundamentals>"
    )
    sections.append(VERIFIED_EVIDENCE_RULE)
    return "\n\n".join(sections)


def _is_present(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())
