"""Tests for trade-proposal coherence and evidence provenance (P1-4, P1-5).

The regressions these guard, all observed in one shipped report:

- Action Sell / entry 520 / stop 540 / sizing "scale out of the existing long":
  a short's stop geometry bolted onto a long's exit.
- That same stop described as "1-1.5x ATR" when ATR was 40.39, making the real
  distance 0.5x — inside the instrument's ordinary daily range.
- "Free cash flow fell 39%" and "capex doubled", both originating in a
  StockTwits post, restated as fact by three downstream agents and used to
  justify an Underweight rating.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from tradingagents.agents.schemas import (
    PositionIntent,
    SentimentBand,
    SentimentReport,
    TraderAction,
    TraderProposal,
    render_sentiment_report,
    render_trader_proposal,
)
from tradingagents.agents.utils.evidence_policy import (
    EVIDENCE_DISCIPLINE,
    UNVERIFIED_MARKER,
)
from tradingagents.dataflows.market_data_validator import (
    TradeReference,
    render_trade_reference_block,
)


def _levels(atr: float = 40.39, close: float = 518.58) -> TradeReference:
    return TradeReference(
        symbol="AMD", as_of="2026-08-04", bar_status="FINAL",
        close=close, atr=atr, ema10=495.02, sma50=514.33, sma200=314.56,
    )


@pytest.mark.unit
class TestPositionIntentCoherence:
    def test_shipped_contradiction_is_rejected(self):
        # The exact combination that shipped: Sell to trim a long, with the stop
        # ABOVE the entry. A stop protecting long exposure must sit below entry.
        with pytest.raises(ValidationError, match="must be BELOW"):
            TraderProposal(
                action=TraderAction.SELL,
                position_intent=PositionIntent.REDUCE_LONG,
                reasoning="Trim into strength.",
                entry_price=520.0,
                stop_loss=540.0,
            )

    def test_same_levels_are_valid_when_the_intent_is_a_short(self):
        # Identical numbers, different intent — this one is coherent.
        proposal = TraderProposal(
            action=TraderAction.SELL,
            position_intent=PositionIntent.OPEN_SHORT,
            reasoning="Open a short into the failed breakout.",
            entry_price=520.0,
            stop_loss=540.0,
        )
        assert proposal.position_intent is PositionIntent.OPEN_SHORT

    def test_long_entry_rejects_a_stop_above_entry(self):
        with pytest.raises(ValidationError, match="must be BELOW"):
            TraderProposal(
                action=TraderAction.BUY,
                position_intent=PositionIntent.OPEN_LONG,
                reasoning="x", entry_price=500.0, stop_loss=510.0,
            )

    def test_short_entry_rejects_a_stop_below_entry(self):
        with pytest.raises(ValidationError, match="must be ABOVE"):
            TraderProposal(
                action=TraderAction.SELL,
                position_intent=PositionIntent.OPEN_SHORT,
                reasoning="x", entry_price=520.0, stop_loss=500.0,
            )

    @pytest.mark.parametrize(
        ("action", "intent"),
        [
            (TraderAction.BUY, PositionIntent.REDUCE_LONG),
            (TraderAction.BUY, PositionIntent.OPEN_SHORT),
            (TraderAction.SELL, PositionIntent.OPEN_LONG),
            (TraderAction.SELL, PositionIntent.NO_CHANGE),
            (TraderAction.HOLD, PositionIntent.OPEN_LONG),
        ],
    )
    def test_action_and_intent_must_agree(self, action, intent):
        with pytest.raises(ValidationError, match="cannot have position_intent"):
            TraderProposal(action=action, position_intent=intent, reasoning="x")

    @pytest.mark.parametrize(
        ("action", "intent"),
        [
            (TraderAction.BUY, PositionIntent.OPEN_LONG),
            (TraderAction.BUY, PositionIntent.REDUCE_SHORT),
            (TraderAction.SELL, PositionIntent.REDUCE_LONG),
            (TraderAction.SELL, PositionIntent.OPEN_SHORT),
            (TraderAction.HOLD, PositionIntent.NO_CHANGE),
        ],
    )
    def test_valid_pairings_are_accepted(self, action, intent):
        assert TraderProposal(
            action=action, position_intent=intent, reasoning="x"
        ).action is action

    def test_missing_levels_skip_the_geometry_check(self):
        # Entry or stop absent is legitimate; only the pairing is checkable.
        assert TraderProposal(
            action=TraderAction.SELL,
            position_intent=PositionIntent.REDUCE_LONG,
            reasoning="x", stop_loss=540.0,
        ).entry_price is None


@pytest.mark.unit
class TestComputedRiskCheck:
    def test_understated_atr_multiple_is_computed_and_flagged(self):
        # 520 entry, 490 stop, ATR 40.39 -> 0.74x ATR. The shipped report called
        # a stop at this distance "1-1.5x ATR".
        proposal = TraderProposal(
            action=TraderAction.SELL,
            position_intent=PositionIntent.REDUCE_LONG,
            reasoning="x", entry_price=520.0, stop_loss=490.0,
        )
        md = render_trader_proposal(proposal, levels=_levels())
        assert "0.74x ATR" in md
        assert "⚠️" in md
        assert "ordinary daily range" in md
        # It also states where a 1.0x ATR stop would actually sit.
        assert "479.61" in md

    def test_adequate_stop_distance_is_not_flagged(self):
        proposal = TraderProposal(
            action=TraderAction.SELL,
            position_intent=PositionIntent.REDUCE_LONG,
            reasoning="x", entry_price=520.0, stop_loss=458.0,
        )
        md = render_trader_proposal(proposal, levels=_levels())
        assert "1.54x ATR" in md
        assert "ordinary daily range" not in md

    def test_short_side_suggests_a_stop_above_entry(self):
        proposal = TraderProposal(
            action=TraderAction.SELL,
            position_intent=PositionIntent.OPEN_SHORT,
            reasoning="x", entry_price=520.0, stop_loss=530.0,
        )
        md = render_trader_proposal(proposal, levels=_levels())
        assert "0.25x ATR" in md
        assert "560.39" in md  # 520 + one ATR, on the correct side

    def test_entry_far_from_the_verified_close_is_flagged(self):
        proposal = TraderProposal(
            action=TraderAction.BUY,
            position_intent=PositionIntent.OPEN_LONG,
            reasoning="x", entry_price=400.0, stop_loss=350.0,
        )
        md = render_trader_proposal(proposal, levels=_levels())
        assert "away from the last verified close" in md

    def test_no_risk_check_without_levels(self):
        proposal = TraderProposal(
            action=TraderAction.BUY,
            position_intent=PositionIntent.OPEN_LONG,
            reasoning="x", entry_price=500.0, stop_loss=460.0,
        )
        assert "Risk Check" not in render_trader_proposal(proposal, levels=None)

    def test_atr_multiple_is_none_when_atr_is_missing(self):
        ref = TradeReference("AMD", "2026-08-04", "FINAL", 518.58, None, None, None, None)
        assert ref.atr_multiple(520.0, 490.0) is None


@pytest.mark.unit
class TestTradeReferenceBlock:
    def test_block_states_levels_and_atr_ladder(self):
        block = render_trade_reference_block(_levels())
        assert "bar status FINAL" in block
        assert "478.19" in block  # 1.0x ATR below close
        assert "458.00" in block  # 1.5x ATR below close

    def test_unavailable_levels_forbid_stating_prices(self):
        block = render_trade_reference_block(None)
        assert "UNAVAILABLE" in block
        assert "Do not state an entry price" in block


@pytest.mark.unit
class TestTraderPromptInjection:
    def test_trader_prompt_carries_verified_levels(self, monkeypatch):
        import tradingagents.agents.trader.trader as trader_mod

        monkeypatch.setattr(
            trader_mod, "get_trade_reference_levels", lambda symbol, date: _levels()
        )
        captured = {}
        proposal = TraderProposal(
            action=TraderAction.SELL,
            position_intent=PositionIntent.REDUCE_LONG,
            reasoning="x", entry_price=520.0, stop_loss=490.0,
        )
        structured = MagicMock()
        structured.invoke.side_effect = lambda p: (
            captured.__setitem__("prompt", p) or proposal
        )
        llm = MagicMock()
        llm.with_structured_output.return_value = structured

        result = trader_mod.create_trader(llm)({
            "company_of_interest": "AMD",
            "trade_date": "2026-08-05",
            "investment_plan": "**Recommendation**: Underweight",
        })

        user_message = captured["prompt"][1]["content"]
        assert "Verified price levels for AMD" in user_message
        assert "40.39" in user_message
        # The computed risk check reaches the saved plan, not just the prompt.
        assert "0.74x ATR" in result["trader_investment_plan"]


@pytest.mark.unit
class TestEvidenceProvenance:
    def test_unverified_claims_are_surfaced_as_their_own_block(self):
        report = SentimentReport(
            overall_band=SentimentBand.MIXED,
            overall_score=4.8,
            confidence="high",
            unverified_numeric_claims=[
                "free cash flow fell 39% — StockTwits poster",
                "capex doubled — StockTwits poster",
            ],
            narrative="Retail is split.",
        )
        md = render_sentiment_report(report)
        assert "Unverified numeric claims from social sources" in md
        assert "free cash flow fell 39%" in md
        assert md.count(UNVERIFIED_MARKER) == 2
        assert "Do not restate them as fact" in md

    def test_marker_is_not_duplicated_when_already_present(self):
        report = SentimentReport(
            overall_band=SentimentBand.BEARISH, overall_score=2.0, confidence="low",
            unverified_numeric_claims=[f"160x P/E {UNVERIFIED_MARKER}"],
            narrative="n",
        )
        assert render_sentiment_report(report).count(UNVERIFIED_MARKER) == 1

    def test_no_block_when_there_are_no_unverified_claims(self):
        report = SentimentReport(
            overall_band=SentimentBand.NEUTRAL, overall_score=5.0,
            confidence="medium", narrative="n",
        )
        md = render_sentiment_report(report)
        assert "Unverified numeric claims" not in md
        assert md.endswith("n")

    @pytest.mark.parametrize(
        "module_path",
        [
            "tradingagents.agents.researchers.bull_researcher",
            "tradingagents.agents.researchers.bear_researcher",
            "tradingagents.agents.risk_mgmt.aggressive_debator",
            "tradingagents.agents.risk_mgmt.conservative_debator",
            "tradingagents.agents.risk_mgmt.neutral_debator",
            "tradingagents.agents.managers.research_manager",
        ],
    )
    def test_every_downstream_agent_imports_the_discipline_rule(self, module_path):
        # An agent that reads the sentiment report but skips this rule is the
        # hole through which an unverified number becomes a stated fact.
        import importlib

        module = importlib.import_module(module_path)
        assert hasattr(module, "get_evidence_discipline_instruction")

    def test_discipline_text_names_all_three_tiers(self):
        for tier in ("Verified", "Reported", "Unverified"):
            assert tier in EVIDENCE_DISCIPLINE
        assert "the verified figure stands" in EVIDENCE_DISCIPLINE
