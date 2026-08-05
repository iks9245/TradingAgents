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
