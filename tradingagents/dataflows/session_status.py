"""Classify whether the latest daily OHLCV bar is settled or still forming.

``load_ohlcv`` deliberately includes the current day's row so an intraday run
sees today's price (#986). Yahoo publishes a *partial* daily candle while the
session is open: its ``Close`` is the last trade, not the settlement price, and
its High/Low/Volume are running totals that will still move. Nothing downstream
could tell that row apart from a settled one, so a snapshot taken at 10:30 ET
would be written up as "the closing price for today" — and a report generated
in another timezone would carry a date label that reads as a completed session
when it is not.

This module answers one question — is the newest bar FINAL or IN-PROGRESS —
from the exchange's own clock, and reports the clock it used so the judgement
is auditable rather than implicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timezone

import pytz

from .symbol_utils import crypto_base, normalize_symbol

BAR_FINAL = "FINAL"
BAR_IN_PROGRESS = "IN-PROGRESS"
BAR_UNKNOWN = "UNKNOWN"

# Minutes after the scheduled close before a daily bar is treated as settled.
# The closing auction print and late-reported trades can still move the official
# close for a few minutes after the bell, so a bar is called FINAL only past
# this buffer. Erring long means an already-settled bar is briefly labelled
# IN-PROGRESS — the harmless direction of the error.
SETTLEMENT_BUFFER_MINUTES = 15

# Exchange clock per instrument class / symbol. ``close`` is the scheduled
# regular-session close in the named timezone.
_US_EQUITY = ("America/New_York", time(16, 0))

# Non-US index symbols the symbol table can resolve to. Without these a Tokyo
# index bar would be judged against the New York clock and flagged IN-PROGRESS
# for most of the US day even though Tokyo settled hours earlier.
_INDEX_SESSIONS: dict[str, tuple[str, time]] = {
    "^GSPC": _US_EQUITY,
    "^NDX": _US_EQUITY,
    "^DJI": _US_EQUITY,
    "^RUT": _US_EQUITY,
    "^VIX": ("America/New_York", time(16, 15)),
    "^GDAXI": ("Europe/Berlin", time(17, 30)),
    "^STOXX50E": ("Europe/Berlin", time(17, 30)),
    "^FCHI": ("Europe/Paris", time(17, 30)),
    "^FTSE": ("Europe/London", time(16, 30)),
    "^N225": ("Asia/Tokyo", time(15, 0)),
    "^HSI": ("Asia/Hong_Kong", time(16, 0)),
}

# Futures and spot FX roll to the next trading day at 17:00 New York, not at
# the equity close.
_FUTURES_FX_SESSION = ("America/New_York", time(17, 0))


@dataclass(frozen=True)
class BarStatus:
    """Whether the newest bar is settled, and the clock that decided it."""

    status: str
    detail: str
    exchange_tz: str
    session_close: str
    as_of_exchange: str
    as_of_utc: str

    @property
    def is_final(self) -> bool:
        return self.status == BAR_FINAL


def _session_for(symbol: str) -> tuple[str, time | None, str]:
    """Return (timezone, scheduled close, instrument-class label) for a symbol.

    A ``None`` close marks a 24-hour instrument, whose daily bar only settles
    at the timezone's midnight rollover.
    """
    canonical = normalize_symbol(symbol or "")

    if crypto_base(canonical):
        # Crypto trades continuously; Yahoo's daily bar rolls at 00:00 UTC.
        return "UTC", None, "crypto (24/7)"
    if canonical in _INDEX_SESSIONS:
        tz_name, close = _INDEX_SESSIONS[canonical]
        return tz_name, close, "index"
    if canonical.endswith("=X"):
        tz_name, close = _FUTURES_FX_SESSION
        return tz_name, close, "spot FX (rolls 17:00 New York)"
    if canonical.endswith("=F"):
        tz_name, close = _FUTURES_FX_SESSION
        return tz_name, close, "futures (rolls 17:00 New York)"

    tz_name, close = _US_EQUITY
    return tz_name, close, "equity (assumed US listing)"


def classify_bar_status(
    latest_bar_date,
    symbol: str,
    now_utc: datetime | None = None,
) -> BarStatus:
    """Decide whether the newest daily bar for ``symbol`` has settled.

    ``latest_bar_date`` is the date of the newest row actually used (anything
    with a ``.date()``, or a ``YYYY-MM-DD`` string). ``now_utc`` is injectable
    so the classification is testable without freezing the system clock.
    """
    now = (now_utc or datetime.now(timezone.utc))
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    now = now.astimezone(timezone.utc)

    tz_name, close, instrument_class = _session_for(symbol)
    tz = pytz.timezone(tz_name)
    now_local = now.astimezone(tz)

    as_of_exchange = now_local.strftime("%Y-%m-%d %H:%M %Z")
    as_of_utc = now.strftime("%Y-%m-%d %H:%M UTC")
    close_label = "00:00 next day (rollover)" if close is None else close.strftime("%H:%M")

    bar_date = _coerce_date(latest_bar_date)
    if bar_date is None:
        return BarStatus(
            BAR_UNKNOWN,
            "Could not parse the bar date, so settlement state is unknown. "
            "Treat this row's Close as provisional.",
            tz_name, close_label, as_of_exchange, as_of_utc,
        )

    def _status(status: str, detail: str) -> BarStatus:
        return BarStatus(status, detail, tz_name, close_label, as_of_exchange, as_of_utc)

    if bar_date < now_local.date():
        return _status(
            BAR_FINAL,
            f"The {bar_date} session closed before the current {tz_name} date "
            f"({now_local.date()}), so this bar is settled.",
        )

    if bar_date > now_local.date():
        return _status(
            BAR_UNKNOWN,
            f"The bar is dated {bar_date}, ahead of the current {tz_name} date "
            f"({now_local.date()}). Treat it as unverified.",
        )

    # The bar is dated today on the exchange's clock.
    if close is None:
        return _status(
            BAR_IN_PROGRESS,
            f"This is a {instrument_class} instrument: the {bar_date} daily bar does not "
            f"settle until the {tz_name} day rolls over. Its Close is the last trade so "
            f"far, and its High/Low/Volume are running totals — not a closing price.",
        )

    settled_at = tz.localize(datetime.combine(bar_date, close)) + _buffer()
    if now_local >= settled_at:
        return _status(
            BAR_FINAL,
            f"The {bar_date} session closed at {close.strftime('%H:%M')} {tz_name} "
            f"(plus a {SETTLEMENT_BUFFER_MINUTES}-minute settlement buffer), which has "
            f"passed. This bar is settled.",
        )

    return _status(
        BAR_IN_PROGRESS,
        f"The {bar_date} session has not settled — it closes at "
        f"{close.strftime('%H:%M')} {tz_name} and it is now "
        f"{now_local.strftime('%H:%M')}. This row's Close is the last trade so far, "
        f"NOT a closing price, and its High/Low/Volume are running totals. Do not "
        f"describe it as '{bar_date} closing price' or as a completed trading day.",
    )


def _buffer():
    from datetime import timedelta
    return timedelta(minutes=SETTLEMENT_BUFFER_MINUTES)


def _coerce_date(value):
    """Best-effort date extraction from a Timestamp, datetime, date, or string."""
    if value is None:
        return None
    if hasattr(value, "date") and callable(value.date):
        try:
            return value.date()
        except Exception:  # noqa: BLE001 — fall through to string parsing
            pass
    if hasattr(value, "year") and hasattr(value, "month") and hasattr(value, "day"):
        return value
    try:
        return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None
