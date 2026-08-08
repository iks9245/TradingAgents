"""The agents after the analysts must be able to check a report claim.

Bull and Bear read the identical four report fields and neither can look
anything up, so a wrong figure used to arrive as a premise both sides argued
*from* rather than *about*. These assert the verified blocks reach the rendered
prompt each of those agents actually sends — not merely that the module
references the helper — and that a state built without them degrades visibly
instead of making a vendor call mid-graph.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager
from tradingagents.agents.managers.research_manager import create_research_manager
from tradingagents.agents.researchers.bear_researcher import create_bear_researcher
from tradingagents.agents.researchers.bull_researcher import create_bull_researcher
from tradingagents.agents.trader.trader import create_trader
from tradingagents.agents.utils.verified_evidence import (
    FUNDAMENTALS_UNAVAILABLE,
    VERIFIED_EVIDENCE_RULE,
    get_verified_evidence_block,
    resolve_verified_evidence,
)
from tradingagents.dataflows.market_data_validator import (
    TradeReference,
    render_trade_reference_block,
)

MARKET_SENTINEL = "MARKET-SNAPSHOT-SENTINEL"
FUNDAMENTALS_SENTINEL = "FUNDAMENTALS-SNAPSHOT-SENTINEL"


def _state(**overrides) -> dict:
    state = {
        "company_of_interest": "NVDA",
        "trade_date": "2026-08-07",
        "asset_type": "stock",
        "verified_market_block": MARKET_SENTINEL,
        "verified_fundamentals_block": FUNDAMENTALS_SENTINEL,
        "market_report": "m",
        "sentiment_report": "s",
        "news_report": "n",
        "fundamentals_report": "f",
        "investment_debate_state": {
            "history": "",
            "bull_history": "",
            "bear_history": "",
            "current_response": "",
            "count": 0,
        },
        "risk_debate_state": {
            "history": "",
            "aggressive_history": "",
            "conservative_history": "",
            "neutral_history": "",
            "latest_speaker": "",
            "current_aggressive_response": "",
            "current_conservative_response": "",
            "current_neutral_response": "",
            "count": 0,
        },
        "investment_plan": "**Recommendation**: Buy",
        "trader_investment_plan": "buy",
    }
    state.update(overrides)
    return state


def _capturing_llm(captured: dict, result=None):
    """LLM recording the prompt handed to both the plain and structured paths."""

    def record(prompt):
        captured["prompt"] = prompt
        return result if result is not None else MagicMock(content="ok")

    structured = MagicMock()
    structured.invoke.side_effect = lambda prompt: (
        captured.__setitem__("prompt", prompt) or result
    )
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    llm.invoke.side_effect = record
    return llm


def _prompt_text(prompt) -> str:
    if isinstance(prompt, str):
        return prompt
    parts = []
    for m in prompt:
        parts.append(m.get("content", "") if isinstance(m, dict) else getattr(m, "content", ""))
    return "\n".join(str(p) for p in parts)


# --- the assembled block -----------------------------------------------------


@pytest.mark.unit
def test_block_carries_both_snapshots_and_the_rule():
    block = get_verified_evidence_block(_state())
    assert MARKET_SENTINEL in block
    assert FUNDAMENTALS_SENTINEL in block
    assert VERIFIED_EVIDENCE_RULE in block


@pytest.mark.unit
def test_include_market_false_omits_levels_but_keeps_fundamentals():
    """The Trader and PM render their own levels block; repeating it would put
    the same numbers in one prompt twice under two headings."""
    block = get_verified_evidence_block(_state(), include_market=False)
    assert MARKET_SENTINEL not in block
    assert FUNDAMENTALS_SENTINEL in block
    assert VERIFIED_EVIDENCE_RULE in block


@pytest.mark.unit
@pytest.mark.parametrize("missing", ["verified_market_block", "verified_fundamentals_block"])
def test_absent_block_degrades_visibly_without_a_vendor_call(missing):
    state = _state(**{missing: ""})
    with patch(
        "tradingagents.agents.utils.verified_evidence.get_trade_reference_levels"
    ) as levels, patch(
        "tradingagents.agents.utils.verified_evidence.render_fundamentals_snapshot_block"
    ) as fundamentals:
        block = get_verified_evidence_block(state)
    levels.assert_not_called()
    fundamentals.assert_not_called()
    assert "UNAVAILABLE" in block


@pytest.mark.unit
def test_rule_tells_the_agent_the_verified_figure_wins():
    """Without this the block is just more context to average into the prose."""
    flat = " ".join(VERIFIED_EVIDENCE_RULE.split())
    assert "the verified figure stands" in flat
    assert "Name the conflict explicitly" in flat


# --- resolution at run start -------------------------------------------------


@pytest.mark.unit
def test_crypto_skips_the_fundamentals_lookup():
    with patch(
        "tradingagents.agents.utils.verified_evidence.get_trade_reference_levels",
        return_value=None,
    ), patch(
        "tradingagents.agents.utils.verified_evidence.render_fundamentals_snapshot_block"
    ) as fundamentals:
        _, block = resolve_verified_evidence("BTC-USD", "2026-08-07", "crypto")
    fundamentals.assert_not_called()
    assert block == FUNDAMENTALS_UNAVAILABLE


@pytest.mark.unit
def test_missing_identifiers_resolve_to_notices_without_a_lookup():
    with patch(
        "tradingagents.agents.utils.verified_evidence.get_trade_reference_levels"
    ) as levels:
        market, fundamentals = resolve_verified_evidence("", "", "stock")
    levels.assert_not_called()
    assert "UNAVAILABLE" in market
    assert fundamentals == FUNDAMENTALS_UNAVAILABLE


# --- the levels renderer's two audiences -------------------------------------


def _ref() -> TradeReference:
    return TradeReference(
        symbol="NVDA",
        as_of="2026-08-07",
        bar_status="FINAL",
        close=100.0,
        atr=4.0,
        ema10=99.0,
        sma50=95.0,
        sma200=90.0,
    )


@pytest.mark.unit
def test_proposal_rule_is_kept_for_the_trader_by_default():
    assert "beneath your proposal" in render_trade_reference_block(_ref())
    # The unavailable notice has always carried its own execution wording.
    assert "leave those fields empty" in render_trade_reference_block(None)


@pytest.mark.unit
@pytest.mark.parametrize("ref", [_ref(), None])
def test_proposal_rule_is_dropped_for_readers_who_set_no_stops(ref):
    """A researcher told its stop arithmetic will be checked beneath a proposal
    it never writes is being described a mechanism that does not apply to it."""
    block = render_trade_reference_block(ref, include_proposal_rule=False)
    assert "beneath your proposal" not in block
    assert "stop" not in block.lower() or "UNAVAILABLE" in block


@pytest.mark.unit
def test_levels_survive_dropping_the_proposal_rule():
    block = render_trade_reference_block(_ref(), include_proposal_rule=False)
    for expected in ("100.00", "4.00", "95.00", "200 SMA"):
        assert expected in block


# --- the blocks reach the prompts each agent actually sends -------------------


@pytest.mark.unit
@pytest.mark.parametrize("factory", [create_bull_researcher, create_bear_researcher])
def test_debaters_are_given_both_snapshots(factory):
    captured = {}
    factory(_capturing_llm(captured))(_state())
    text = _prompt_text(captured["prompt"])
    assert MARKET_SENTINEL in text
    assert FUNDAMENTALS_SENTINEL in text


@pytest.mark.unit
def test_research_manager_is_given_both_snapshots():
    """It adjudicates on the debate transcript alone — it never sees the
    analyst reports, so the snapshots are its only reference."""
    from tradingagents.agents.schemas import PortfolioRating, ResearchPlan

    captured = {}
    llm = _capturing_llm(
        captured,
        ResearchPlan(
            recommendation=PortfolioRating.HOLD,
            rationale="x",
            strategic_actions="hold",
        ),
    )
    create_research_manager(llm)(_state())
    text = _prompt_text(captured["prompt"])
    assert MARKET_SENTINEL in text
    assert FUNDAMENTALS_SENTINEL in text


@pytest.mark.unit
def test_trader_is_given_the_fundamentals_snapshot():
    from tradingagents.agents.schemas import (
        PositionIntent,
        TraderAction,
        TraderProposal,
    )

    captured = {}
    llm = _capturing_llm(
        captured,
        TraderProposal(
            action=TraderAction.BUY,
            position_intent=PositionIntent.OPEN_LONG,
            reasoning="x",
        ),
    )
    create_trader(llm)(_state())
    assert FUNDAMENTALS_SENTINEL in _prompt_text(captured["prompt"])


@pytest.mark.unit
def test_portfolio_manager_is_given_the_fundamentals_snapshot():
    """It set a target justified as "40x 2025 PE" when FY EPS made it 170x.
    Price levels cannot check a multiple; the EPS behind one lives here."""
    from tradingagents.agents.schemas import PortfolioDecision, PortfolioRating

    captured = {}
    llm = _capturing_llm(
        captured,
        PortfolioDecision(
            rating=PortfolioRating.HOLD,
            executive_summary="x",
            investment_thesis="x",
        ),
    )
    create_portfolio_manager(llm)(_state())
    assert FUNDAMENTALS_SENTINEL in _prompt_text(captured["prompt"])


@pytest.mark.unit
def test_portfolio_manager_no_longer_claims_it_lacks_fundamentals():
    """The prompt used to say it could not derive a multiple because it had no
    fundamentals. It has them now, and a prompt that contradicts its own
    context teaches the model to ignore one of the two."""
    import inspect

    import tradingagents.agents.managers.portfolio_manager as pm

    src = inspect.getsource(pm)
    assert "You do not have\nthe fundamentals report in this prompt" not in src
    assert "do not have the fundamentals report" not in src.lower().replace("\n", " ")
