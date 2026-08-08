"""Deterministic market-data verification snapshot.

The market analyst is an LLM that can confabulate exact numbers — citing a
Bollinger band or a "historically validated bounce" that the underlying data
doesn't support (#830). This module computes a ground-truth snapshot (latest
OHLCV row on or before the analysis date, common indicators, recent closes)
the analyst is told to treat as the source of truth for any exact numeric
claim. Deterministic, no LLM involved.
"""

from __future__ import annotations

import functools
from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd
from stockstats import wrap

from tradingagents.dataflows.session_status import BAR_FINAL, classify_bar_status
from tradingagents.dataflows.stockstats_utils import load_ohlcv

# A fixed, common indicator set so the snapshot is the same shape every run.
DEFAULT_SNAPSHOT_INDICATORS: tuple[str, ...] = (
    "close_10_ema", "close_50_sma", "close_200_sma",
    "rsi", "boll", "boll_ub", "boll_lb",
    "macd", "macds", "macdh", "atr",
)


def _verified_rows(symbol: str, curr_date: str) -> pd.DataFrame:
    """OHLCV on or before curr_date, date-sorted. Raises if nothing usable.

    ``load_ohlcv`` already normalizes the Date column and filters out
    look-ahead rows, but we re-apply the cutoff defensively — this is a
    verification path, so it must not trust its input to be pre-filtered.
    """
    data = load_ohlcv(symbol, curr_date)
    if data is None or data.empty:
        raise ValueError(f"No OHLCV data available for {symbol}.")

    df = data.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df[df["Date"] <= pd.to_datetime(curr_date)].sort_values("Date")
    if df.empty:
        raise ValueError(f"No OHLCV rows on or before {curr_date} for {symbol}.")
    return df


def _fmt(value) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int,)):
        return str(value)
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


@dataclass(frozen=True)
class TradeReference:
    """Price levels the Trader and Portfolio Manager need to size a trade.

    The Trader runs without tools — it only sees the Research Manager's prose
    plan — so it has historically invented entry and stop levels with no idea
    what the instrument actually costs or how far it moves in a day. One
    shipped proposal set a stop 20 points from entry on an instrument whose ATR
    was 40, then described it as "1-1.5x ATR": it was 0.5x, a stop that normal
    daily noise would take out. These are the numbers needed to check that
    arithmetic, resolved once and injected into the prompt.
    """

    symbol: str
    as_of: str
    bar_status: str
    close: float | None
    atr: float | None
    ema10: float | None
    sma50: float | None
    sma200: float | None
    week52_high: float | None = None
    week52_low: float | None = None

    def atr_multiple(self, entry: float | None, stop: float | None) -> float | None:
        """How many ATRs separate an entry from its stop, or None if incomputable."""
        if entry is None or stop is None or not self.atr:
            return None
        return abs(entry - stop) / self.atr


@functools.lru_cache(maxsize=64)
def get_trade_reference_levels(symbol: str, curr_date: str) -> TradeReference | None:
    """Resolve the price levels behind a trade proposal, or None when unavailable.

    Returns None rather than raising: a missing snapshot must degrade the
    Trader's prompt to "no verified levels available", never block the run.

    Cached because three call sites now want the same levels within one run —
    the run-start resolution that fills the state, the Trader, and the
    Portfolio Manager — and the settled bars behind them cannot change while a
    run is in flight. ``TradeReference`` is frozen, so sharing one instance
    between callers is safe.
    """
    if not symbol or not curr_date:
        return None
    try:
        df = _verified_rows(symbol, curr_date)
        stock_df = wrap(df.copy())

        def _indicator(name: str) -> float | None:
            try:
                stock_df[name]  # triggers stockstats calculation
                value = stock_df.iloc[-1][name]
                return None if pd.isna(value) else float(value)
            except Exception:  # noqa: BLE001 — one missing indicator is not fatal
                return None

        latest = df.iloc[-1]
        close = latest.get("Close")
        bar = classify_bar_status(latest["Date"], symbol)

        # 52-week extremes come from the same settled OHLCV frame as every other
        # level here. A vendor's own 52-week fields are computed on a different
        # schedule from a different adjustment basis, so mixing the two puts two
        # numbers for one statistic in the same report.
        year = df[df["Date"] > latest["Date"] - pd.Timedelta(days=365)]
        high = year["High"].max() if "High" in year.columns and not year.empty else None
        low = year["Low"].min() if "Low" in year.columns and not year.empty else None

        return TradeReference(
            symbol=symbol.upper(),
            as_of=_fmt(latest["Date"]),
            bar_status=bar.status,
            close=None if pd.isna(close) else float(close),
            atr=_indicator("atr"),
            ema10=_indicator("close_10_ema"),
            sma50=_indicator("close_50_sma"),
            sma200=_indicator("close_200_sma"),
            week52_high=None if high is None or pd.isna(high) else float(high),
            week52_low=None if low is None or pd.isna(low) else float(low),
        )
    except Exception:  # noqa: BLE001 — the caller renders an explicit unavailable note
        return None


def render_trade_reference_block(
    ref: TradeReference | None, *, include_proposal_rule: bool = True
) -> str:
    """Render trade levels for a prompt, or an explicit unavailable notice.

    ``include_proposal_rule=False`` drops the closing paragraph about stop
    distances being checked against the proposal. Researchers and the Research
    Manager read these levels to test claims in the reports; they do not set
    stops, and telling them their arithmetic will be checked beneath a proposal
    they never write describes a mechanism that does not apply to them.
    """
    if ref is None:
        if not include_proposal_rule:
            return (
                "**Verified price levels: UNAVAILABLE.** No market snapshot could "
                "be resolved for this instrument. Do not treat any price level, "
                "moving average, or volatility figure in the reports as verified."
            )
        return (
            "**Verified price levels: UNAVAILABLE.** No market snapshot could be "
            "resolved for this instrument. Do not state an entry price, a stop "
            "loss, or any ATR multiple — say that levels could not be verified "
            "and leave those fields empty."
        )

    def _line(label: str, value: float | None) -> str:
        return f"- {label}: {value:,.2f}" if value is not None else f"- {label}: N/A"

    lines = [
        f"**Verified price levels for {ref.symbol}** "
        f"(as of {ref.as_of}, bar status {ref.bar_status}):",
        _line("Last close", ref.close),
        _line("ATR (daily volatility)", ref.atr),
        _line("10 EMA", ref.ema10),
        _line("50 SMA", ref.sma50),
        _line("200 SMA", ref.sma200),
        _line("52-week high", ref.week52_high),
        _line("52-week low", ref.week52_low),
    ]
    if ref.atr and ref.close:
        lines += [
            f"- 1.0x ATR below last close: {ref.close - ref.atr:,.2f}",
            f"- 1.5x ATR below last close: {ref.close - 1.5 * ref.atr:,.2f}",
            f"- 1.0x ATR above last close: {ref.close + ref.atr:,.2f}",
            f"- 1.5x ATR above last close: {ref.close + 1.5 * ref.atr:,.2f}",
        ]
    if include_proposal_rule:
        lines += [
            "",
            "Every price level you state must be consistent with these numbers. If you "
            "describe a stop as a multiple of ATR, it must match the distance you "
            "actually set — the arithmetic is checked and any mismatch is printed "
            "beneath your proposal.",
        ]
    else:
        lines += [
            "",
            "These are the settled figures for this instrument. Any price level, "
            "moving average, or volatility figure stated in the reports or in the "
            "debate must be consistent with them.",
        ]
    return "\n".join(lines)


def build_verified_market_snapshot(
    symbol: str,
    curr_date: str,
    look_back_days: int = 30,
    indicators: Iterable[str] | None = None,
) -> str:
    """Render a ground-truth snapshot: latest OHLCV row, indicators, recent closes."""
    # `df` keeps the original capitalized OHLCV columns (Open/High/Low/Close/
    # Volume); stockstats `wrap()` lowercases columns and adds indicator
    # columns, so read raw prices from `df` and indicators from `stock_df`.
    df = _verified_rows(symbol, curr_date)
    stock_df = wrap(df.copy())

    selected = tuple(indicators or DEFAULT_SNAPSHOT_INDICATORS)
    indicator_values: dict[str, str] = {}
    for name in selected:
        try:
            stock_df[name]  # triggers stockstats calculation
            indicator_values[name] = _fmt(stock_df.iloc[-1][name])
        except Exception as exc:  # noqa: BLE001 — one bad indicator shouldn't sink the snapshot
            indicator_values[name] = f"N/A ({type(exc).__name__})"

    latest = df.iloc[-1]
    latest_date = _fmt(latest["Date"])
    window = max(1, min(int(look_back_days), 30))
    recent = df.tail(window)

    # The newest row may be a partial candle for a session that is still open.
    # Say so explicitly: without it, an intraday Close reads as a closing price
    # and the row's date reads as a completed trading day.
    bar = classify_bar_status(latest["Date"], symbol)
    close_label = "Close" if bar.is_final else "Close (last trade so far — NOT a closing price)"

    lines = [
        f"## Verified market data snapshot for {symbol.upper()}",
        "",
        f"- Requested analysis date: {curr_date}",
        f"- Latest trading row used: {latest_date}",
        f"- **Bar status: {bar.status}** — {bar.detail}",
        f"- Snapshot taken at: {bar.as_of_exchange} / {bar.as_of_utc}",
        f"- Exchange clock assumed: {bar.exchange_tz}, session close {bar.session_close}",
        "- Rows after the requested analysis date are excluded before verification.",
        "",
        f"### Latest verified OHLCV row ({latest_date}, {bar.status})",
        "",
        "| Field | Value |",
        "|---|---:|",
    ]
    for field in ("Open", "High", "Low", "Close", "Volume"):
        label = close_label if field == "Close" else field
        lines.append(f"| {label} | {_fmt(latest.get(field))} |")
    if not bar.is_final:
        lines += [
            "",
            f"> The {latest_date} row above is **not a settled session**. Its High, Low, "
            "and Volume are running totals that will still move, and every indicator "
            "below is computed with this unsettled bar as its last input. Label any "
            f"figure taken from it as an intraday reading as of {bar.as_of_exchange} — "
            f"never as the {latest_date} close, high, low, or volume.",
        ]

    lines += ["", "### Verified technical indicators (latest row)", "",
              "| Indicator | Value |", "|---|---:|"]
    for name, value in indicator_values.items():
        lines.append(f"| {name} | {value} |")

    lines += ["", f"### Recent verified closes (last {len(recent)} rows)", "",
              "| Date | Close |", "|---|---:|"]
    for _, row in recent.iterrows():
        lines.append(f"| {_fmt(row['Date'])} | {_fmt(row.get('Close'))} |")

    lines += [
        "",
        "Use this snapshot as the source of truth for exact OHLCV, price-level, "
        "and indicator-value claims. If another tool output conflicts with it, "
        "flag the discrepancy rather than inventing a reconciled number. Do not "
        "claim historical validation, support/resistance bounces, or exact "
        "percentage moves unless directly supported by tool output with concrete "
        "dates and prices. Every price you attribute to a date must carry the "
        "bar status above: only a "
        f"{BAR_FINAL} bar may be called a close, a session high, or a session low.",
    ]
    return "\n".join(lines)
