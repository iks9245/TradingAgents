from typing import Annotated

from langchain_core.tools import tool

from tradingagents.dataflows.fundamentals_validator import (
    build_verified_fundamentals_snapshot,
)


@tool
def get_verified_fundamentals_snapshot(
    ticker: Annotated[str, "ticker symbol of the company"],
    curr_date: Annotated[str, "the current trading date, YYYY-mm-dd"],
) -> str:
    """Deterministic verification snapshot for exact fundamental claims.

    Returns date-safe income-statement, balance-sheet, cash-flow, and valuation
    calculations with their operands shown. Call this before making exact claims
    about margins, growth, leverage, liquidity, free cash flow, EPS, or P/E,
    and treat it as the source of truth.
    """
    return build_verified_fundamentals_snapshot(ticker, curr_date)
