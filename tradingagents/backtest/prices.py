"""Price history cache and forward-return computation for the backtest harness.

Two things this module exists to get right:

**Batching.** A backtest evaluates hundreds of (ticker, date) points but touches
only a handful of distinct symbols. Fetching per decision would mean hundreds of
yfinance calls and near-certain rate limiting. :class:`PriceCache` fetches each
symbol's full history once, covering the whole backtest span, then slices it per
evaluation point.

**Entry timing.** ``entry_offset`` controls which bar the position is opened on,
relative to the decision date. The default of 1 (the *next* trading day's close)
is the defensible backtest convention: an analysis run against data through the
decision date cannot also be executed at that same date's close. Setting it to 0
reproduces the same-close convention used by the live memory log
(``TradingAgentsGraph._fetch_returns``), which is fine for a rough reflection
signal but flatters a backtest.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

from tradingagents.dataflows.stockstats_utils import yf_retry
from tradingagents.dataflows.symbol_utils import normalize_symbol

logger = logging.getLogger(__name__)

# Calendar-day padding either side of the requested span. The tail must cover the
# longest holding period plus weekends/holidays; the head keeps a little slack so
# a decision date landing on a market holiday still finds a following bar.
_TAIL_PAD_DAYS = 30
_HEAD_PAD_DAYS = 5


@dataclass(frozen=True)
class ForwardReturn:
    """Realized outcome of holding ``ticker`` from ``entry_date`` to ``exit_date``."""

    ticker: str
    decision_date: str
    entry_date: str
    exit_date: str
    holding_days: int
    raw_return: float
    benchmark_return: float

    @property
    def alpha_return(self) -> float:
        """Excess return over the benchmark across the identical bar window."""
        return self.raw_return - self.benchmark_return


class MissingPriceData(Exception):
    """Raised when a symbol has no usable bars for a requested window."""


class PriceCache:
    """Fetch-once, slice-many price history for a fixed set of symbols.

    ``prefetch`` is the only network path; every ``forward_return`` call after it
    is pure pandas. Symbols that fail to load are recorded and raise
    :class:`MissingPriceData` on use, so one delisted ticker degrades that
    ticker's rows rather than aborting the whole backtest.
    """

    def __init__(self, *, auto_adjust: bool = True):
        # auto_adjust folds splits and dividends into the close, which is what a
        # total-return backtest wants; a split inside the holding window would
        # otherwise register as a ~-50% "loss".
        self._auto_adjust = auto_adjust
        self._frames: dict[str, pd.DataFrame] = {}
        self._failed: dict[str, str] = {}

    def prefetch(self, symbols, start: str, end: str) -> None:
        """Load daily history for every symbol covering ``start``..``end``.

        Symbols are normalized through :func:`normalize_symbol` so broker-style
        input (``XAUUSD``, ``BTCUSD``) resolves to the same instrument the
        analysis priced. Padding is applied so the last decision date still has
        a full holding window of bars after it.
        """
        span_start = (datetime.strptime(start, "%Y-%m-%d") - timedelta(days=_HEAD_PAD_DAYS))
        span_end = (datetime.strptime(end, "%Y-%m-%d") + timedelta(days=_TAIL_PAD_DAYS))

        for symbol in dict.fromkeys(symbols):  # de-dup, preserve order
            if symbol in self._frames or symbol in self._failed:
                continue
            canonical = normalize_symbol(symbol)
            try:
                frame = yf_retry(
                    lambda c=canonical: yf.Ticker(c).history(
                        start=span_start.strftime("%Y-%m-%d"),
                        end=span_end.strftime("%Y-%m-%d"),
                        auto_adjust=self._auto_adjust,
                    )
                )
            except Exception as exc:  # network, rate limit exhaustion, bad symbol
                logger.warning("Price history unavailable for %s (%s): %s", symbol, canonical, exc)
                self._failed[symbol] = str(exc)
                continue

            if frame is None or frame.empty or "Close" not in frame.columns:
                logger.warning("Price history empty for %s (%s)", symbol, canonical)
                self._failed[symbol] = "empty history"
                continue

            self._frames[symbol] = _normalize_frame(frame)

    def load_frame(self, symbol: str, frame: pd.DataFrame) -> None:
        """Install price history for ``symbol`` directly, bypassing the vendor.

        For backtesting against your own data (a CSV of adjusted closes, a
        survivorship-free vendor extract) and for tests that must not touch the
        network. ``frame`` needs a ``Close`` column and a date-like index.
        """
        if "Close" not in frame.columns:
            raise ValueError(f"{symbol}: frame has no 'Close' column")
        if frame.empty:
            raise ValueError(f"{symbol}: frame is empty")
        self._frames[symbol] = _normalize_frame(frame)
        self._failed.pop(symbol, None)

    def load_csv_dir(self, directory: str | Path, symbols) -> None:
        """Load ``<SYMBOL>.csv`` from ``directory`` for each requested symbol.

        Each file needs a date column (``Date``, ``date``, or the index) and a
        ``Close`` column. Use this to backtest against a survivorship-free vendor
        extract, to make runs reproducible independently of what the vendor
        currently serves, or to run in a sandbox with no network access.
        """
        directory = Path(directory).expanduser()
        for symbol in dict.fromkeys(symbols):
            path = directory / f"{symbol}.csv"
            if not path.exists():
                self._failed[symbol] = f"no CSV at {path}"
                continue
            try:
                frame = pd.read_csv(path)
                date_col = next(
                    (c for c in ("Date", "date", "Datetime", "timestamp") if c in frame.columns),
                    None,
                )
                if date_col is None:
                    raise ValueError("no date column (expected one of Date/date/Datetime/timestamp)")
                frame = frame.set_index(pd.to_datetime(frame[date_col])).drop(columns=[date_col])
                self.load_frame(symbol, frame)
            except Exception as exc:
                logger.warning("Could not load %s: %s", path, exc)
                self._failed[symbol] = str(exc)

    def loaded_symbols(self) -> list[str]:
        """Symbols with usable history, in insertion order."""
        return list(self._frames)

    def failures(self) -> dict[str, str]:
        """Symbol -> reason for every symbol that could not be loaded."""
        return dict(self._failed)

    def _frame(self, symbol: str) -> pd.DataFrame:
        if symbol in self._frames:
            return self._frames[symbol]
        reason = self._failed.get(symbol, "not prefetched")
        raise MissingPriceData(f"no price history for {symbol}: {reason}")

    def forward_return(
        self,
        ticker: str,
        decision_date: str,
        holding_days: int,
        *,
        benchmark: str,
        entry_offset: int = 1,
    ) -> ForwardReturn:
        """Return the realized outcome of holding ``ticker`` after ``decision_date``.

        Entry and exit bars are located on the *ticker's* calendar and the
        benchmark is then measured across the same dates, so a paired comparison
        is not distorted by one instrument trading on a day the other does not
        (e.g. crypto over a weekend against an equity benchmark).

        Raises :class:`MissingPriceData` when either leg lacks the bars needed to
        cover the full window — a truncated window is silently shorter exposure,
        which would quietly bias results toward zero.
        """
        if holding_days < 1:
            raise ValueError(f"holding_days must be >= 1, got {holding_days}")
        if entry_offset < 0:
            raise ValueError(f"entry_offset must be >= 0, got {entry_offset}")

        frame = self._frame(ticker)
        bench_frame = self._frame(benchmark)

        entry_pos = _first_bar_at_or_after(frame, decision_date)
        if entry_pos is None:
            raise MissingPriceData(f"{ticker}: no bar on or after {decision_date}")
        entry_pos += entry_offset
        exit_pos = entry_pos + holding_days
        if exit_pos >= len(frame):
            raise MissingPriceData(
                f"{ticker}: only {len(frame) - entry_pos} bars after entry, "
                f"need {holding_days + 1} for a {holding_days}-day hold from {decision_date}"
            )

        entry_date = frame.index[entry_pos]
        exit_date = frame.index[exit_pos]
        raw = _pct_change(frame, entry_pos, exit_pos)
        bench = _return_between_dates(bench_frame, entry_date, exit_date, benchmark)

        return ForwardReturn(
            ticker=ticker,
            decision_date=decision_date,
            entry_date=entry_date.strftime("%Y-%m-%d"),
            exit_date=exit_date.strftime("%Y-%m-%d"),
            holding_days=holding_days,
            raw_return=raw,
            benchmark_return=bench,
        )


def _normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Sort by date and drop tz/intraday detail so index lookups are date-exact."""
    frame = frame.copy()
    index = pd.to_datetime(frame.index)
    # yfinance returns tz-aware timestamps for most exchanges. Comparing those
    # against naive decision dates raises, and the tz offset can shift the
    # calendar date, so normalize to naive midnight in the exchange's own tz.
    if getattr(index, "tz", None) is not None:
        index = index.tz_localize(None)
    frame.index = index.normalize()
    frame = frame[~frame.index.duplicated(keep="last")]
    return frame.sort_index()


def _first_bar_at_or_after(frame: pd.DataFrame, date: str) -> int | None:
    """Index position of the first bar on or after ``date``, or None."""
    target = pd.Timestamp(date).normalize()
    pos = frame.index.searchsorted(target, side="left")
    if pos >= len(frame):
        return None
    return int(pos)


def _pct_change(frame: pd.DataFrame, start_pos: int, end_pos: int) -> float:
    start_price = float(frame["Close"].iloc[start_pos])
    end_price = float(frame["Close"].iloc[end_pos])
    if start_price <= 0:
        raise MissingPriceData(f"non-positive entry price {start_price} at position {start_pos}")
    return (end_price - start_price) / start_price


def _return_between_dates(
    frame: pd.DataFrame, entry_date: pd.Timestamp, exit_date: pd.Timestamp, label: str
) -> float:
    """Benchmark return measured on the ticker's entry/exit dates.

    Uses the last benchmark bar at or before each date so a benchmark holiday
    does not drop the row; requires the two to resolve to distinct bars.
    """
    entry_pos = frame.index.searchsorted(entry_date, side="right") - 1
    exit_pos = frame.index.searchsorted(exit_date, side="right") - 1
    if entry_pos < 0 or exit_pos < 0:
        raise MissingPriceData(f"{label}: no bars covering {entry_date.date()}..{exit_date.date()}")
    if exit_pos <= entry_pos:
        raise MissingPriceData(
            f"{label}: entry and exit resolve to the same bar for "
            f"{entry_date.date()}..{exit_date.date()}"
        )
    return _pct_change(frame, int(entry_pos), int(exit_pos))
