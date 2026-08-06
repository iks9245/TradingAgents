"""The OHLCV cache must never serve a partial bar as settled history (#1150).

Yahoo publishes an in-progress daily candle during market hours, and row
inspection cannot tell it from a final one. What decides whether a cached file
is usable is therefore not "is the requested date before today" but "was this
file written after the requested session settled".

Those two questions have the same answer only in the Americas. On a Taipei
clock at 06:26 on 6 August, a request for 5 August looked like history although
New York had closed barely two hours earlier — so a cache written during that
session was served as settled, and a report published 100.42 as Intel's 5 August
close when the settled close was 101.06. These tests pin the settlement-based
rule and the timezone case that broke the calendar-based one.
"""
from __future__ import annotations

import os
import time

import pandas as pd
import pytest

import tradingagents.dataflows.stockstats_utils as su
from tradingagents.dataflows.session_status import session_settled_at

TODAY = pd.Timestamp("2026-07-18")
STALE = su.OHLCV_CACHE_TTL_SECONDS + 60


def _write(tmp_path, name="cache.csv", written_at=None, age_seconds=0.0, last_date="2026-07-17"):
    """Create a cache file whose mtime is a chosen instant (or a chosen age)."""
    f = tmp_path / name
    pd.DataFrame({"Date": [last_date], "Close": [1.0]}).to_csv(f, index=False)
    if written_at is not None:
        stamp = pd.Timestamp(written_at).timestamp()
        os.utime(f, (stamp, stamp))
    elif age_seconds:
        old = time.time() - age_seconds
        os.utime(f, (old, old))
    return str(f)


def _settled(date: str, symbol: str = "AAPL") -> pd.Timestamp:
    return pd.Timestamp(session_settled_at(pd.Timestamp(date), symbol))


@pytest.mark.unit
class TestSettlementDecidesReuse:
    def test_cache_written_during_the_session_is_refetched(self, tmp_path):
        # 11:00 in New York on the requested day: the bar is still forming.
        f = _write(tmp_path, written_at=_settled("2026-07-18") - pd.Timedelta(hours=5))
        assert su._needs_same_day_refresh(f, TODAY, TODAY, "AAPL") is True

    def test_cache_written_after_the_close_is_reused(self, tmp_path):
        f = _write(tmp_path, written_at=_settled("2026-07-18") + pd.Timedelta(minutes=1))
        assert su._needs_same_day_refresh(f, TODAY, TODAY, "AAPL") is False

    def test_cache_predating_the_requested_day_is_refetched(self, tmp_path):
        # It cannot contain the requested session's rows at all.
        f = _write(tmp_path, written_at=pd.Timestamp("2026-07-15 12:00"))
        assert su._needs_same_day_refresh(f, TODAY, TODAY, "AAPL") is True

    def test_historical_request_reuses_a_cache_written_since(self, tmp_path):
        # Past sessions are immutable once the file postdates them.
        past = pd.Timestamp("2026-05-01")
        f = _write(tmp_path, age_seconds=STALE, last_date="2026-04-30")
        assert su._needs_same_day_refresh(f, past, TODAY, "AAPL") is False

    def test_recent_unsettled_cache_is_not_hammered(self, tmp_path):
        # Written moments ago and the session has not settled: the TTL still
        # applies, so repeated runs (or a weekend) cannot spam the vendor.
        # The requested day is deliberately ahead of the clock so settlement is
        # unambiguously in the future whenever this test runs.
        future = pd.Timestamp.today().normalize() + pd.Timedelta(days=2)
        f = _write(tmp_path)
        assert su._needs_same_day_refresh(f, future, TODAY, "AAPL") is False


@pytest.mark.unit
class TestHostTimezoneNoLongerDecides:
    """The regression: a host clock east of the Americas.

    At 06:26 Taipei on 6 August the host calendar says "5 August is history",
    but New York settled that session only ~2 hours earlier — so a cache written
    during it held a partial bar.
    """

    REQUESTED = pd.Timestamp("2026-08-05")
    TAIPEI_TODAY = pd.Timestamp("2026-08-06 06:26")

    def test_mid_session_cache_is_refetched_despite_looking_historical(self, tmp_path):
        f = _write(tmp_path, written_at=pd.Timestamp("2026-08-05 15:00"))  # 11:00 ET
        assert su._needs_same_day_refresh(f, self.REQUESTED, self.TAIPEI_TODAY, "INTC") is True

    def test_post_close_cache_is_reused(self, tmp_path):
        f = _write(tmp_path, written_at=pd.Timestamp("2026-08-05 21:00"))  # 17:00 ET
        assert su._needs_same_day_refresh(f, self.REQUESTED, self.TAIPEI_TODAY, "INTC") is False

    def test_a_crypto_day_only_settles_at_utc_midnight(self, tmp_path):
        # A 24/7 instrument's current-day bar is never final mid-day.
        f = _write(tmp_path, written_at=pd.Timestamp("2026-08-06 12:00"))
        assert su._needs_same_day_refresh(
            f, pd.Timestamp("2026-08-06"), self.TAIPEI_TODAY, "BTC-USD"
        ) is True

    def test_unparseable_date_falls_back_to_the_calendar_rule(self, tmp_path):
        f = _write(tmp_path, age_seconds=STALE)
        assert su._needs_same_day_refresh(f, pd.NaT, TODAY, "AAPL") is False


@pytest.mark.unit
def test_load_ohlcv_refetches_a_cache_written_before_settlement(tmp_path, monkeypatch):
    """End-to-end: the helper is actually wired into load_ohlcv's cache branch.

    Without this, the unit tests above would still pass if the helper were never
    called from the real code path.
    """
    monkeypatch.setattr(su, "get_config", lambda: {"data_cache_dir": str(tmp_path)})
    monkeypatch.setattr(su.pd.Timestamp, "today", staticmethod(lambda: TODAY))

    start = (TODAY - pd.DateOffset(years=5)).strftime("%Y-%m-%d")
    end = (TODAY + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    cache_file = tmp_path / f"AAPL-YFin-data-{start}-{end}.csv"
    pd.DataFrame({"Date": ["2026-07-17"], "Close": [100.0]}).to_csv(cache_file, index=False)
    mid_session = (_settled("2026-07-18") - pd.Timedelta(hours=5)).timestamp()
    os.utime(cache_file, (mid_session, mid_session))

    calls = []

    def _fake_download(*a, **k):
        calls.append(1)
        return pd.DataFrame(
            {"Date": pd.to_datetime(["2026-07-17", "2026-07-18"]), "Close": [100.0, 222.0]}
        ).set_index("Date")

    monkeypatch.setattr(su.yf, "download", _fake_download)

    out = su.load_ohlcv("AAPL", TODAY.strftime("%Y-%m-%d"))

    assert calls, "a cache written before settlement must trigger a refetch"
    assert 222.0 in out["Close"].values, "refreshed close must reach the caller"


@pytest.mark.unit
def test_load_ohlcv_reuses_a_cache_written_after_settlement(tmp_path, monkeypatch):
    # Mirror image: a settled cache must NOT trigger a download.
    monkeypatch.setattr(su, "get_config", lambda: {"data_cache_dir": str(tmp_path)})
    monkeypatch.setattr(su.pd.Timestamp, "today", staticmethod(lambda: TODAY))

    start = (TODAY - pd.DateOffset(years=5)).strftime("%Y-%m-%d")
    end = (TODAY + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    cache_file = tmp_path / f"AAPL-YFin-data-{start}-{end}.csv"
    pd.DataFrame({"Date": ["2026-07-18"], "Close": [100.0]}).to_csv(cache_file, index=False)
    settled = (_settled("2026-07-18") + pd.Timedelta(minutes=1)).timestamp()
    os.utime(cache_file, (settled, settled))

    def _fail_download(*a, **k):
        raise AssertionError("a settled cache must not refetch")

    monkeypatch.setattr(su.yf, "download", _fail_download)
    su.load_ohlcv("AAPL", TODAY.strftime("%Y-%m-%d"))
