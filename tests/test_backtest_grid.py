"""Grid construction: settlement clamping, cutoff flagging, universe resolution."""

from datetime import date

import pytest

from tradingagents.backtest.grid import build_grid, latest_settled_date
from tradingagents.backtest.universe import UNIVERSES, resolve_universe

TODAY = date(2026, 6, 1)


@pytest.mark.unit
def test_grid_is_ticker_by_date_product():
    grid = build_grid(
        ["AAPL", "MSFT"], start="2025-01-01", end="2025-03-01",
        holding_days=21, step_days=21, today=TODAY,
    )
    assert grid.tickers == ("AAPL", "MSFT")
    assert len(grid) == len(grid.dates) * 2
    # Date-major ordering: an interrupted run covers whole dates, not whole tickers.
    assert grid.points[0].date == grid.points[1].date
    assert grid.points[0].ticker != grid.points[1].ticker


@pytest.mark.unit
def test_unsettled_dates_are_clamped_out():
    """A decision whose holding window has not closed has no realized return."""
    grid = build_grid(
        ["AAPL"], start="2026-01-01", end="2026-06-01",
        holding_days=21, step_days=7, today=TODAY,
    )
    assert grid.end <= latest_settled_date(21, today=TODAY)
    # ~29 trading days of slack for a 21-day hold, so late May must be excluded.
    assert grid.end < "2026-05-15"


@pytest.mark.unit
def test_fully_unsettled_range_raises_rather_than_returning_empty():
    with pytest.raises(ValueError, match="no settled decision dates"):
        build_grid(["AAPL"], start="2026-05-25", end="2026-06-01", holding_days=21, today=TODAY)


@pytest.mark.unit
def test_knowledge_cutoff_flags_but_does_not_drop():
    grid = build_grid(
        ["AAPL"], start="2025-01-01", end="2025-06-01",
        holding_days=5, step_days=30, knowledge_cutoff="2025-03-01", today=TODAY,
    )
    assert grid.contaminated_count > 0
    assert len(grid.clean()) > 0
    # Flagged, not filtered — the gap between subsets is the interesting number.
    assert len(grid) == grid.contaminated_count + len(grid.clean())
    assert all(p.date <= "2025-03-01" for p in grid.points if p.contaminated)


@pytest.mark.unit
def test_cutoff_boundary_date_counts_as_contaminated():
    grid = build_grid(
        ["AAPL"], start="2025-03-03", end="2025-03-03",
        holding_days=5, knowledge_cutoff="2025-03-03", today=TODAY,
    )
    assert grid.points[0].contaminated is True


@pytest.mark.unit
def test_decision_dates_never_land_on_a_weekend():
    grid = build_grid(
        ["AAPL"], start="2025-01-04", end="2025-04-01",
        holding_days=5, step_days=7, today=TODAY,
    )
    import pandas as pd

    assert all(pd.Timestamp(d).weekday() < 5 for d in grid.dates)
    assert len(grid.dates) == len(set(grid.dates))


@pytest.mark.unit
def test_grid_requires_tickers():
    with pytest.raises(ValueError, match="at least one ticker"):
        build_grid([], start="2025-01-01", today=TODAY)


@pytest.mark.unit
def test_point_key_is_stable_and_unique():
    grid = build_grid(["AAPL", "MSFT"], start="2025-01-01", end="2025-03-01", today=TODAY)
    keys = [p.key for p in grid]
    assert len(keys) == len(set(keys))
    assert keys[0] == f"{grid.points[0].ticker}@{grid.points[0].date}"


@pytest.mark.unit
@pytest.mark.parametrize("name", sorted(UNIVERSES))
def test_named_universes_resolve_to_nonempty_tuples(name):
    tickers = resolve_universe(name)
    assert tickers and all(isinstance(t, str) and t for t in tickers)


@pytest.mark.unit
def test_universe_accepts_comma_list_and_explicit_sequence():
    assert resolve_universe("aapl, msft") == ("AAPL", "MSFT")
    assert resolve_universe(["aapl", "msft"]) == ("AAPL", "MSFT")


@pytest.mark.unit
def test_empty_universe_string_raises():
    with pytest.raises(ValueError, match="empty universe"):
        resolve_universe("  ,  ")
