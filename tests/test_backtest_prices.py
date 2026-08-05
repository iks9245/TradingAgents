"""Forward-return computation: entry timing, alpha pairing, and missing data.

All offline — frames are installed via ``PriceCache.load_frame`` so nothing here
touches yfinance.
"""

import pandas as pd
import pytest

from tradingagents.backtest.prices import MissingPriceData, PriceCache


def _frame(prices, start="2025-01-01", freq="B"):
    """Business-day frame whose closes are ``prices``."""
    index = pd.bdate_range(start=start, periods=len(prices)) if freq == "B" else pd.date_range(
        start=start, periods=len(prices), freq=freq
    )
    return pd.DataFrame({"Close": [float(p) for p in prices]}, index=index)


@pytest.fixture()
def cache():
    c = PriceCache()
    # Ticker rises 1% per bar; benchmark is flat, so alpha == raw.
    c.load_frame("UP", _frame([100 * (1.01 ** i) for i in range(40)]))
    c.load_frame("FLAT", _frame([100.0] * 40))
    return c


@pytest.mark.unit
def test_entry_offset_defaults_to_the_next_bar(cache):
    """An analysis run through the decision date cannot execute at that close."""
    at_close = cache.forward_return("UP", "2025-01-01", 5, benchmark="FLAT", entry_offset=0)
    next_day = cache.forward_return("UP", "2025-01-01", 5, benchmark="FLAT")

    assert at_close.entry_date == "2025-01-01"
    assert next_day.entry_date == "2025-01-02"
    # Same holding length either way; only the window shifts.
    assert at_close.holding_days == next_day.holding_days == 5
    assert next_day.raw_return == pytest.approx(at_close.raw_return)


@pytest.mark.unit
def test_holding_window_spans_exactly_n_trading_days(cache):
    result = cache.forward_return("UP", "2025-01-01", 5, benchmark="FLAT", entry_offset=0)
    assert result.entry_date == "2025-01-01"
    assert result.exit_date == "2025-01-08"  # 5 business days later
    assert result.raw_return == pytest.approx(1.01 ** 5 - 1)


@pytest.mark.unit
def test_alpha_is_raw_minus_benchmark_over_the_same_window(cache):
    cache.load_frame("BENCH", _frame([100 * (1.005 ** i) for i in range(40)]))
    result = cache.forward_return("UP", "2025-01-01", 10, benchmark="BENCH", entry_offset=0)

    assert result.benchmark_return == pytest.approx(1.005 ** 10 - 1)
    assert result.alpha_return == pytest.approx(result.raw_return - result.benchmark_return)
    assert result.alpha_return > 0


@pytest.mark.unit
def test_decision_on_a_market_holiday_rolls_to_the_next_bar(cache):
    # 2025-01-04 is a Saturday; entry must land on the following Monday's bar.
    result = cache.forward_return("UP", "2025-01-04", 3, benchmark="FLAT", entry_offset=0)
    assert result.entry_date == "2025-01-06"


@pytest.mark.unit
def test_truncated_window_raises_instead_of_shortening_the_hold(cache):
    """A short window is reduced exposure, which would bias returns toward zero."""
    with pytest.raises(MissingPriceData, match="bars after entry"):
        cache.forward_return("UP", "2025-02-20", 21, benchmark="FLAT")


@pytest.mark.unit
def test_unknown_symbol_raises_missing_price_data(cache):
    with pytest.raises(MissingPriceData, match="no price history"):
        cache.forward_return("NOPE", "2025-01-02", 5, benchmark="FLAT")


@pytest.mark.unit
def test_benchmark_uses_last_bar_at_or_before_each_date():
    """A benchmark holiday must not drop the row; it falls back to the prior bar."""
    cache = PriceCache()
    cache.load_frame("CRYPTO", _frame([100 * (1.01 ** i) for i in range(30)], freq="D"))
    cache.load_frame("SPY", _frame([100 * (1.002 ** i) for i in range(30)]))

    # Entry lands on a weekend bar the equity benchmark does not have.
    result = cache.forward_return("CRYPTO", "2025-01-04", 7, benchmark="SPY", entry_offset=0)
    assert result.entry_date == "2025-01-04"
    assert result.benchmark_return != 0.0


@pytest.mark.unit
def test_tz_aware_index_is_normalized_to_naive_dates():
    cache = PriceCache()
    index = pd.bdate_range(start="2025-01-01", periods=20, tz="America/New_York")
    cache.load_frame("TZ", pd.DataFrame({"Close": [100.0 + i for i in range(20)]}, index=index))

    result = cache.forward_return("TZ", "2025-01-02", 3, benchmark="TZ", entry_offset=0)
    assert result.entry_date == "2025-01-02"
    # Same instrument on both legs: alpha is exactly zero, proving the two legs
    # resolved to identical bars rather than shifting by a timezone offset.
    assert result.alpha_return == pytest.approx(0.0)


@pytest.mark.unit
def test_invalid_arguments_are_rejected(cache):
    with pytest.raises(ValueError, match="holding_days"):
        cache.forward_return("UP", "2025-01-02", 0, benchmark="FLAT")
    with pytest.raises(ValueError, match="entry_offset"):
        cache.forward_return("UP", "2025-01-02", 5, benchmark="FLAT", entry_offset=-1)


@pytest.mark.unit
def test_load_frame_rejects_unusable_input():
    cache = PriceCache()
    with pytest.raises(ValueError, match="no 'Close'"):
        cache.load_frame("X", pd.DataFrame({"Open": [1.0]}, index=pd.bdate_range("2025-01-01", periods=1)))
    with pytest.raises(ValueError, match="empty"):
        cache.load_frame("X", pd.DataFrame({"Close": []}))
