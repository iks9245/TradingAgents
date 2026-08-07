"""Tests for deterministic, unit-explicit fundamental verification."""

from __future__ import annotations

import pandas as pd
import pytest

import tradingagents.dataflows.fundamentals_validator as validator


def _frame(rows: dict[str, list[float]], columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(rows, index=pd.to_datetime(columns)).T


def _ticker() -> object:
    annual_columns = ["2025-12-31", "2024-12-31", "2023-12-31", "2022-12-31", "2026-12-31"]
    quarterly_columns = ["2025-03-31", "2024-03-31", "2024-12-31", "2024-09-30", "2024-06-30"]

    class FakeTicker:
        income_stmt = _frame({
            "Total Revenue": [346_390_000, 300_000_000, 250_000_000, 200_000_000, 999_000_000],
            "Gross Profit": [171_520_000, 150_000_000, 120_000_000, 100_000_000, 999_000_000],
            "Operating Income": [36_940_000, 30_000_000, 25_000_000, 20_000_000, 999_000_000],
            "Net Income": [43_350_000, 35_000_000, 30_000_000, -10_000_000, 999_000_000],
            "Diluted EPS": [2.65, 2.00, 1.50, 1.00, 99.00],
        }, annual_columns)
        quarterly_income_stmt = _frame({"Diluted EPS": [0.80, 0.70, 0.75, 0.72, 0.65]}, quarterly_columns)
        balance_sheet = pd.DataFrame()
        quarterly_balance_sheet = _frame({
            "Total Assets": [100_000_000], "Total Liabilities": [20_000_000],
            "Stockholders Equity": [64_462_000], "Total Debt": [3_871_000],
            "Long Term Debt": [2_997_000], "Current Assets": [26_834_000],
            "Current Liabilities": [9_829_000], "Inventory": [8_045_000],
        }, ["2025-03-31"])
        cashflow = _frame({
            "Operating Cash Flow": [20_000_000, 10_000_000, 9_000_000, 8_000_000, 999_000_000],
            "Capital Expenditure": [-5_000_000, -4_000_000, -3_000_000, -2_000_000, -999_000_000],
            "Free Cash Flow": [15_000_000, 6_000_000, 6_000_000, 6_000_000, 999_000_000],
        }, annual_columns)
        quarterly_cashflow = _frame({
            "Operating Cash Flow": [6_000_000, 3_000_000, 4_000_000, 3_500_000, 3_200_000],
            "Capital Expenditure": [-2_000_000, -1_000_000, -1_100_000, -1_200_000, -1_300_000],
            "Free Cash Flow": [4_000_000, 2_000_000, 2_900_000, 2_300_000, 1_900_000],
        }, quarterly_columns)

    return FakeTicker()


@pytest.mark.unit
class TestVerifiedFundamentalsSnapshot:
    def test_ratios_arithmetic_and_future_columns(self, monkeypatch):
        fake = _ticker()
        monkeypatch.setattr(validator.yf, "Ticker", lambda symbol: fake)
        snapshot = validator.build_verified_fundamentals_snapshot("AMD", "2025-12-31", reference_price=450.0)

        assert "10.66%  (37 / 346)" in snapshot
        assert "12.51%  (43 / 346)" in snapshot
        assert "6.01%  (= 0.0601x)  (4 / 64)" in snapshot
        assert "169.81x" in snapshot
        assert "999" not in snapshot

    def test_missing_line_item_and_non_positive_growth_degrade_gracefully(self, monkeypatch):
        fake = _ticker()
        fake.quarterly_balance_sheet = pd.DataFrame()
        fake.income_stmt.loc["Operating Income", pd.Timestamp("2024-12-31")] = -1.0
        monkeypatch.setattr(validator.yf, "Ticker", lambda symbol: fake)
        snapshot = validator.build_verified_fundamentals_snapshot("AMD", "2025-12-31", reference_price=450.0)

        assert "Balance sheet, most recent quarter" not in snapshot
        assert "N/M" in snapshot

    def test_tool_delegates_to_builder(self, monkeypatch):
        fake = _ticker()
        monkeypatch.setattr(validator.yf, "Ticker", lambda symbol: fake)
        monkeypatch.setattr(validator, "load_ohlcv", lambda symbol, date: pd.DataFrame())
        from tradingagents.agents.utils.fundamental_validation_tools import (
            get_verified_fundamentals_snapshot,
        )

        snapshot = get_verified_fundamentals_snapshot.invoke({"ticker": "AMD", "curr_date": "2025-12-31"})
        assert "Verified fundamentals snapshot for AMD" in snapshot


@pytest.mark.unit
class TestSnapshotIsInjectedNotOffered:
    """The snapshot must reach the analyst without the model choosing to fetch it.

    On the 2026-08-06 INTC run it was bound as a tool, the model never called
    it, and every ratio in the report came from the raw vendor dump instead.
    """

    def _node_prompt(self, monkeypatch):
        from unittest.mock import MagicMock

        from langchain_core.messages import AIMessage
        from langchain_core.runnables import RunnableLambda

        import tradingagents.agents.analysts.fundamentals_analyst as fa

        captured = {}
        llm = MagicMock()
        llm.bind_tools.side_effect = lambda tools: (
            captured.__setitem__("tools", [t.name for t in tools])
            or RunnableLambda(lambda p: captured.__setitem__("prompt", p) or AIMessage(content="ok"))
        )
        fa.create_fundamentals_analyst(llm)({
            "trade_date": "2025-12-31", "company_of_interest": "AMD", "messages": [],
        })
        return captured

    def test_snapshot_text_is_in_the_system_message(self, monkeypatch):
        fake = _ticker()
        monkeypatch.setattr(validator.yf, "Ticker", lambda symbol: fake)
        monkeypatch.setattr(validator, "load_ohlcv", lambda symbol, date: pd.DataFrame())
        validator.render_fundamentals_snapshot_block.cache_clear()

        captured = self._node_prompt(monkeypatch)
        system_message = captured["prompt"].messages[0].content
        assert "<start_of_verified_fundamentals>" in system_message
        assert "Verified fundamentals snapshot for AMD" in system_message
        assert "no tool to call" in system_message

    def test_snapshot_is_no_longer_a_tool_the_model_can_skip(self, monkeypatch):
        fake = _ticker()
        monkeypatch.setattr(validator.yf, "Ticker", lambda symbol: fake)
        monkeypatch.setattr(validator, "load_ohlcv", lambda symbol, date: pd.DataFrame())
        validator.render_fundamentals_snapshot_block.cache_clear()

        assert "get_verified_fundamentals_snapshot" not in self._node_prompt(monkeypatch)["tools"]

    def test_failure_degrades_to_an_explicit_notice(self, monkeypatch):
        def boom(*args, **kwargs):
            raise RuntimeError("vendor down")

        monkeypatch.setattr(validator, "build_verified_fundamentals_snapshot", boom)
        validator.render_fundamentals_snapshot_block.cache_clear()

        block = validator.render_fundamentals_snapshot_block("AMD", "2025-12-31")
        assert "UNAVAILABLE" in block
        # The analyst must be told not to invent the figures it cannot verify.
        assert "do not state a margin" in block.lower()

    def test_missing_arguments_do_not_reach_the_vendor(self, monkeypatch):
        called = []
        monkeypatch.setattr(
            validator, "build_verified_fundamentals_snapshot",
            lambda *a, **k: called.append(a) or "x",
        )
        validator.render_fundamentals_snapshot_block.cache_clear()

        assert "UNAVAILABLE" in validator.render_fundamentals_snapshot_block("", "2025-12-31")
        assert "UNAVAILABLE" in validator.render_fundamentals_snapshot_block("AMD", "")
        assert called == []

    def test_repeated_calls_hit_the_cache(self, monkeypatch):
        # The node re-runs on every turn of its tool loop; without caching that
        # is a fresh round of vendor calls each time.
        calls = []
        monkeypatch.setattr(
            validator, "build_verified_fundamentals_snapshot",
            lambda s, d: calls.append((s, d)) or "snapshot",
        )
        validator.render_fundamentals_snapshot_block.cache_clear()

        for _ in range(3):
            validator.render_fundamentals_snapshot_block("AMD", "2025-12-31")
        assert len(calls) == 1


@pytest.mark.unit
class TestOperatingIncomeCrossCheck:
    """The vendor's operating income must reconcile with its own expense lines.

    yfinance reports INTC 2026 Q2 operating income as 1,966M while itemising
    expenses that give 1,805M — the 161M restructuring charge is listed but not
    deducted. That inflated figure reached a shipped report as "operating income
    swung to $1.97B" and became the bull case's central evidence.
    """

    def _frame_with(self, **rows) -> pd.DataFrame:
        return _frame({k.replace("_", " "): [v] for k, v in rows.items()}, ["2026-06-30"])

    def _snapshot(self, monkeypatch, quarterly: pd.DataFrame) -> str:
        fake = _ticker()
        fake.quarterly_income_stmt = quarterly
        fake.income_stmt = pd.DataFrame()
        monkeypatch.setattr(validator.yf, "Ticker", lambda symbol: fake)
        monkeypatch.setattr(validator, "load_ohlcv", lambda s, d: pd.DataFrame())
        return validator.build_verified_fundamentals_snapshot("INTC", "2026-06-30")

    def test_excluded_restructuring_is_flagged_with_both_values(self, monkeypatch):
        quarterly = _frame({
            "Total Revenue": [16_128_000_000.0], "Gross Profit": [6_509_000_000.0],
            "Research And Development": [3_368_000_000.0],
            "Selling General And Administration": [1_175_000_000.0],
            "Restructuring And Mergern Acquisition": [161_000_000.0],
            "Operating Income": [1_966_000_000.0],
        }, ["2026-06-30"])
        snapshot = self._snapshot(monkeypatch, quarterly)
        assert "⚠️ MISMATCH" in snapshot
        assert "1,966" in snapshot and "1,805" in snapshot
        assert "restructuring/M&A line (161)" in snapshot
        assert "do not build a turnaround narrative on it" in snapshot

    def test_a_consistent_statement_is_not_flagged(self, monkeypatch):
        quarterly = _frame({
            "Total Revenue": [16_128_000_000.0], "Gross Profit": [6_509_000_000.0],
            "Research And Development": [3_368_000_000.0],
            "Selling General And Administration": [1_175_000_000.0],
            "Restructuring And Mergern Acquisition": [161_000_000.0],
            "Operating Income": [1_805_000_000.0],
        }, ["2026-06-30"])
        snapshot = self._snapshot(monkeypatch, quarterly)
        assert "⚠️ MISMATCH" not in snapshot
        assert "agree for every period shown" in snapshot

    def test_unknown_expense_rows_do_not_blame_the_vendor(self, monkeypatch):
        # When our recomputation comes out LOWER than reported the vendor is
        # inflating; when it comes out HIGHER we simply do not know every
        # expense row it uses (AMD's intangibles amortization). Only the first
        # is the vendor's fault, so only the first is reported.
        quarterly = _frame({
            "Total Revenue": [16_128_000_000.0], "Gross Profit": [6_509_000_000.0],
            "Research And Development": [3_368_000_000.0],
            "Selling General And Administration": [1_175_000_000.0],
            "Operating Income": [1_000_000_000.0],
        }, ["2026-06-30"])
        snapshot = self._snapshot(monkeypatch, quarterly)
        assert "⚠️ MISMATCH" not in snapshot

    def test_section_is_skipped_when_the_rows_are_absent(self, monkeypatch):
        snapshot = self._snapshot(monkeypatch, _frame({"Diluted EPS": [0.8]}, ["2026-06-30"]))
        assert "Operating income cross-check" not in snapshot


@pytest.mark.unit
class TestVendorRatioAndSignChecks:
    """Vendor ratio periods and sign/label traps, from the 2026-08-06 INTC report."""

    def _snapshot(self, monkeypatch, quarterly: pd.DataFrame, info: dict) -> str:
        fake = _ticker()
        fake.quarterly_income_stmt = quarterly
        fake.income_stmt = pd.DataFrame()
        fake.info = info
        monkeypatch.setattr(validator.yf, "Ticker", lambda symbol: fake)
        monkeypatch.setattr(validator, "load_ohlcv", lambda s, d: pd.DataFrame())
        return validator.build_verified_fundamentals_snapshot("INTC", "2026-06-30")

    def _four_quarters(self) -> pd.DataFrame:
        # Revenue and operating income chosen so the two windows differ sharply:
        # latest quarter 12.19%, trailing four quarters 7.55% — Intel's real gap.
        return _frame({
            "Total Revenue": [16_128e6, 13_577e6, 13_674e6, 13_653e6],
            "Operating Income": [1_966e6, 934e6, 550e6, 858e6],
            "Gross Profit": [6_509e6, 5_347e6, 4_943e6, 5_218e6],
            "Net Income": [-11_033e6, -3_728e6, -591e6, 4_063e6],
        }, ["2026-06-30", "2026-03-31", "2025-12-31", "2025-09-30"])

    def test_a_single_quarter_ratio_is_not_called_trailing_twelve_months(self, monkeypatch):
        snapshot = self._snapshot(
            monkeypatch, self._four_quarters(), {"operatingMargins": 0.1219}
        )
        assert "most recent quarter" in snapshot
        # Both windows are shown, so the 4.6pp gap is visible rather than implied.
        assert "12.19%" in snapshot and "7.55%" in snapshot

    def test_a_trailing_ratio_is_identified_as_trailing(self, monkeypatch):
        snapshot = self._snapshot(
            monkeypatch, self._four_quarters(), {"profitMargins": -0.1979}
        )
        assert "trailing 12 months" in snapshot

    def test_a_ratio_matching_neither_window_is_flagged(self, monkeypatch):
        snapshot = self._snapshot(
            monkeypatch, self._four_quarters(), {"operatingMargins": 0.42}
        )
        assert "⚠️ neither" in snapshot
        assert "does not reproduce either window" in snapshot

    def test_a_negative_gain_row_is_reported_as_a_loss(self, monkeypatch):
        quarterly = _frame({
            "Total Revenue": [16_128e6], "Gross Profit": [6_509e6],
            "Gain On Sale Of Security": [-12_476e6],
        }, ["2026-06-30"])
        snapshot = self._snapshot(monkeypatch, quarterly, {})
        assert "Sign-versus-label contradictions" in snapshot
        assert "a LOSS of 12,476" in snapshot
        assert "it is part of the loss" in snapshot

    def test_a_positive_gain_row_is_left_alone(self, monkeypatch):
        quarterly = _frame({
            "Total Revenue": [16_128e6], "Gross Profit": [6_509e6],
            "Gain On Sale Of Security": [1_200e6],
        }, ["2026-06-30"])
        assert "Sign-versus-label contradictions" not in self._snapshot(monkeypatch, quarterly, {})

    def test_cash_flow_values_carry_their_own_labels(self, monkeypatch):
        # A value lifted out of a table loses its column; these lines do not.
        fake = _ticker()
        monkeypatch.setattr(validator.yf, "Ticker", lambda symbol: fake)
        monkeypatch.setattr(validator, "load_ohlcv", lambda s, d: pd.DataFrame())
        snapshot = validator.build_verified_fundamentals_snapshot("AMD", "2025-12-31")
        assert "labelled series" in snapshot
        assert "OCF" in snapshot and "FCF" in snapshot
        assert "Never restate one as the other" in snapshot


@pytest.mark.unit
class TestQuarterlyMarginsAndFcfDefinition:
    """Margin deltas in points, and an FCF figure that names its definition."""

    def _snapshot(self, monkeypatch) -> str:
        fake = _ticker()
        # Intel's real Q1/Q2 2026 figures: the QoQ gross-margin move a shipped
        # report called "+2.3 percentage points" is actually +0.98.
        fake.quarterly_income_stmt = _frame({
            "Total Revenue": [16_128e6, 13_577e6],
            "Gross Profit": [6_509e6, 5_347e6],
            "Operating Income": [1_966e6, 934e6],
            "Net Income": [-11_033e6, -3_728e6],
        }, ["2026-06-30", "2026-03-31"])
        fake.income_stmt = pd.DataFrame()
        monkeypatch.setattr(validator.yf, "Ticker", lambda symbol: fake)
        monkeypatch.setattr(validator, "load_ohlcv", lambda s, d: pd.DataFrame())
        return validator.build_verified_fundamentals_snapshot("INTC", "2026-06-30")

    def test_quarterly_margins_are_computed_with_their_operands(self, monkeypatch):
        snapshot = self._snapshot(monkeypatch)
        assert "40.36%  (6,509 / 16,128)" in snapshot
        assert "39.38%  (5,347 / 13,577)" in snapshot

    def test_the_qoq_change_is_stated_in_points_not_guessed(self, monkeypatch):
        snapshot = self._snapshot(monkeypatch)
        assert "+0.98 pp  (40.36% - 39.38%)" in snapshot
        # The figure the report invented must not appear.
        assert "2.3 pp" not in snapshot

    def test_points_and_percent_are_distinguished(self, monkeypatch):
        snapshot = self._snapshot(monkeypatch)
        assert "percentage points" in snapshot
        assert "the relative change" in snapshot
        # ...and quarterly margins must not be compared against other windows.
        assert "Do not compare a quarterly margin against an annual or TTM margin" in snapshot

    def test_free_cash_flow_names_its_definition(self, monkeypatch):
        fake = _ticker()
        monkeypatch.setattr(validator.yf, "Ticker", lambda symbol: fake)
        monkeypatch.setattr(validator, "load_ohlcv", lambda s, d: pd.DataFrame())
        snapshot = validator.build_verified_fundamentals_snapshot("AMD", "2025-12-31")
        assert "Simplified FCF (OCF - capex)" in snapshot
        assert "simplified FCF (OCF - capex)" in snapshot  # the labelled series too

    def test_the_company_defined_measure_is_flagged_as_absent(self, monkeypatch):
        fake = _ticker()
        monkeypatch.setattr(validator.yf, "Ticker", lambda symbol: fake)
        monkeypatch.setattr(validator, "load_ohlcv", lambda s, d: pd.DataFrame())
        snapshot = validator.build_verified_fundamentals_snapshot("AMD", "2025-12-31")
        assert "adjusted free cash flow" in snapshot
        assert "opposite" in snapshot
        # The specific over-claim this exists to prevent.
        assert "self-funded" in snapshot
