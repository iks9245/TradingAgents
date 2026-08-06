"""Tests for daily-bar settlement classification (P0-3).

The regression these guard: ``load_ohlcv`` includes the current day's row, which
Yahoo publishes as a partial candle while the session is open. Nothing marked
that row as unsettled, so an intraday snapshot was written up as a completed
trading day's close, high, low, and volume.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tradingagents.dataflows.session_status import (
    BAR_FINAL,
    BAR_IN_PROGRESS,
    BAR_UNKNOWN,
    classify_bar_status,
)


def _utc(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)


@pytest.mark.unit
class TestUSEquitySessions:
    def test_prior_day_bar_is_final(self):
        status = classify_bar_status("2026-08-04", "AMD", _utc("2026-08-05 11:31"))
        assert status.status == BAR_FINAL
        assert status.is_final

    def test_bar_dated_today_during_the_session_is_in_progress(self):
        # 2026-08-05 14:00 UTC is 10:00 in New York — the session is open.
        status = classify_bar_status("2026-08-05", "AMD", _utc("2026-08-05 14:00"))
        assert status.status == BAR_IN_PROGRESS
        assert "NOT a closing price" in status.detail
        assert "running totals" in status.detail

    def test_bar_dated_today_after_the_close_is_final(self):
        # 21:00 UTC is 17:00 New York, past the 16:00 close plus the buffer.
        status = classify_bar_status("2026-08-05", "AMD", _utc("2026-08-05 21:00"))
        assert status.status == BAR_FINAL

    def test_settlement_buffer_keeps_the_bar_provisional_just_after_the_bell(self):
        # 20:05 UTC is 16:05 New York: the bell has rung but the closing auction
        # print can still move the official close, so FINAL would be premature.
        status = classify_bar_status("2026-08-05", "AMD", _utc("2026-08-05 20:05"))
        assert status.status == BAR_IN_PROGRESS

    def test_premarket_run_treats_a_today_dated_bar_as_unsettled(self):
        # 11:31 UTC is 07:31 New York — before the open. This is the exact clock
        # position of the AMD report that labelled an unsettled row as a close.
        status = classify_bar_status("2026-08-05", "AMD", _utc("2026-08-05 11:31"))
        assert status.status == BAR_IN_PROGRESS

    def test_reports_the_clock_it_used(self):
        status = classify_bar_status("2026-08-04", "AMD", _utc("2026-08-05 11:31"))
        assert status.exchange_tz == "America/New_York"
        assert status.session_close == "16:00"
        # Both clocks are shown so a reader in any timezone can audit the call.
        assert "2026-08-05" in status.as_of_utc
        assert "UTC" in status.as_of_utc
        assert status.as_of_exchange


@pytest.mark.unit
class TestOtherInstrumentClasses:
    def test_crypto_bar_dated_today_never_settles_until_rollover(self):
        status = classify_bar_status("2026-08-05", "BTCUSD", _utc("2026-08-05 23:59"))
        assert status.status == BAR_IN_PROGRESS
        assert status.exchange_tz == "UTC"
        assert "24/7" in status.detail

    def test_crypto_prior_day_is_final(self):
        status = classify_bar_status("2026-08-04", "BTC-USD", _utc("2026-08-05 00:30"))
        assert status.status == BAR_FINAL

    def test_tokyo_index_is_judged_on_the_tokyo_clock(self):
        # 08:00 UTC is 17:00 in Tokyo — past the 15:00 close. Judged on the New
        # York clock this would wrongly read as an unsettled bar.
        status = classify_bar_status("2026-08-05", "JP225", _utc("2026-08-05 08:00"))
        assert status.status == BAR_FINAL
        assert status.exchange_tz == "Asia/Tokyo"

    def test_futures_roll_at_1700_new_york(self):
        status = classify_bar_status("2026-08-05", "XAUUSD+", _utc("2026-08-05 21:00"))
        assert status.exchange_tz == "America/New_York"
        assert status.session_close == "17:00"
        assert status.status == BAR_IN_PROGRESS  # 17:00 NY, buffer not yet elapsed


@pytest.mark.unit
class TestDegradedInputs:
    def test_unparseable_date_is_unknown_not_final(self):
        status = classify_bar_status("not-a-date", "AMD", _utc("2026-08-05 21:00"))
        assert status.status == BAR_UNKNOWN
        assert not status.is_final

    def test_future_dated_bar_is_unknown(self):
        status = classify_bar_status("2026-08-09", "AMD", _utc("2026-08-05 21:00"))
        assert status.status == BAR_UNKNOWN

    def test_naive_now_is_treated_as_utc(self):
        naive = datetime(2026, 8, 5, 14, 0)
        assert classify_bar_status("2026-08-05", "AMD", naive).status == BAR_IN_PROGRESS


@pytest.mark.unit
class TestSessionSettledAt:
    """The settlement instant, used to decide whether a cached file is stale."""

    def test_us_equity_settles_after_the_close_plus_buffer(self):
        from tradingagents.dataflows.session_status import (
            SETTLEMENT_BUFFER_MINUTES,
            session_settled_at,
        )

        settled = session_settled_at("2026-08-05", "INTC")
        # 16:00 New York on 2026-08-05 is 20:00 UTC; plus the buffer.
        assert settled == _utc("2026-08-05 20:00") + timedelta(minutes=SETTLEMENT_BUFFER_MINUTES)

    def test_crypto_settles_at_the_next_utc_midnight(self):
        from tradingagents.dataflows.session_status import session_settled_at

        assert session_settled_at("2026-08-05", "BTC-USD") == _utc("2026-08-06 00:00")

    def test_tokyo_settles_on_the_tokyo_clock(self):
        from tradingagents.dataflows.session_status import (
            SETTLEMENT_BUFFER_MINUTES,
            session_settled_at,
        )

        # 15:00 Tokyo on 2026-08-05 is 06:00 UTC.
        assert session_settled_at("2026-08-05", "JP225") == _utc(
            "2026-08-05 06:00"
        ) + timedelta(minutes=SETTLEMENT_BUFFER_MINUTES)

    def test_unparseable_date_returns_none(self):
        from tradingagents.dataflows.session_status import session_settled_at

        assert session_settled_at("not-a-date", "INTC") is None

    def test_settlement_agrees_with_the_status_classifier(self):
        # One clock, two views: a bar is FINAL exactly when "now" has passed the
        # settlement instant. Drift between them would let a cache be judged
        # stale while its bar is called settled.
        from tradingagents.dataflows.session_status import (
            BAR_FINAL,
            classify_bar_status,
            session_settled_at,
        )

        settled = session_settled_at("2026-08-05", "INTC")
        assert classify_bar_status("2026-08-05", "INTC", settled).status == BAR_FINAL
        just_before = settled - timedelta(minutes=1)
        assert classify_bar_status("2026-08-05", "INTC", just_before).status != BAR_FINAL
