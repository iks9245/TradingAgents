"""Evaluation grid: the fixed set of (ticker, date) points every strategy sees.

The grid is built once and shared by every strategy in a run. That is what makes
the comparison *paired*: the LLM pipeline and the random baseline are scored on
identical decision points, so the difference between them cannot be explained by
one having drawn an easier sample of dates or tickers.

Two guards are enforced here rather than left to the caller:

**Settlement.** A decision date whose holding window has not finished yet has no
realized return. Including it would silently drop rows later, and those drops are
not random — the most recent (often most volatile) points go missing.

**Knowledge contamination.** An LLM's weights encode what happened after any date
before its training cutoff. A "backtest" over such dates measures recall, not
prediction. Points at or before ``knowledge_cutoff`` are flagged so results can
be split; they are not silently dropped, because the *size* of the gap between
contaminated and clean subsets is itself the interesting number.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta

import pandas as pd

# Trading days per calendar week, used to convert a holding period in trading
# days into the calendar slack needed before a decision can have settled.
_CALENDAR_PER_TRADING_DAY = 7 / 5
# Extra calendar days on top, absorbing holiday clusters and the lag before a
# freshly closed bar is available from the data vendor.
_SETTLE_SLACK_DAYS = 10


@dataclass(frozen=True)
class EvalPoint:
    """One decision the strategies are asked to make."""

    ticker: str
    date: str
    contaminated: bool = False

    @property
    def key(self) -> str:
        """Stable identifier used to key the decision cache."""
        return f"{self.ticker}@{self.date}"


@dataclass(frozen=True)
class EvaluationGrid:
    """An ordered set of evaluation points plus the span they cover."""

    points: tuple[EvalPoint, ...]
    holding_days: int
    knowledge_cutoff: str | None = None

    def __len__(self) -> int:
        return len(self.points)

    def __iter__(self):
        return iter(self.points)

    @property
    def tickers(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(p.ticker for p in self.points))

    @property
    def dates(self) -> tuple[str, ...]:
        return tuple(sorted({p.date for p in self.points}))

    @property
    def start(self) -> str:
        return self.dates[0]

    @property
    def end(self) -> str:
        return self.dates[-1]

    @property
    def contaminated_count(self) -> int:
        return sum(1 for p in self.points if p.contaminated)

    def clean(self) -> tuple[EvalPoint, ...]:
        """Points strictly after the knowledge cutoff — the only out-of-sample ones."""
        return tuple(p for p in self.points if not p.contaminated)


def latest_settled_date(holding_days: int, *, today: date | None = None) -> str:
    """Latest decision date whose ``holding_days`` window has certainly closed."""
    today = today or datetime.now().date()
    slack = int(holding_days * _CALENDAR_PER_TRADING_DAY) + _SETTLE_SLACK_DAYS
    return (today - timedelta(days=slack)).strftime("%Y-%m-%d")


def build_grid(
    tickers,
    *,
    start: str,
    end: str | None = None,
    holding_days: int = 21,
    step_days: int = 21,
    knowledge_cutoff: str | None = None,
    today: date | None = None,
) -> EvaluationGrid:
    """Build a rolling (ticker, date) grid.

    Args:
        tickers: instruments to evaluate; every date is applied to every ticker.
        start: first decision date (inclusive). Snapped forward to a weekday.
        end: last decision date. Defaults to — and is always clamped to — the
            latest date whose holding window has settled.
        holding_days: trading days each position is held. 21 (~one month) by
            default: 5-day windows on single names are dominated by noise, so a
            short horizon needs far more samples to resolve any real signal.
        step_days: calendar days between consecutive decision dates. Keep this
            at or above ``holding_days`` to avoid overlapping windows, which
            correlate observations and inflate apparent significance.
        knowledge_cutoff: model training cutoff, ``YYYY-MM-DD``. Points at or
            before it are flagged contaminated.
        today: injectable clock for tests.

    Raises:
        ValueError: if the settled window is empty or no tickers were given.
    """
    tickers = tuple(dict.fromkeys(t.strip().upper() for t in tickers if t and t.strip()))
    if not tickers:
        raise ValueError("build_grid requires at least one ticker")
    if step_days < 1:
        raise ValueError(f"step_days must be >= 1, got {step_days}")

    settled_end = latest_settled_date(holding_days, today=today)
    effective_end = min(end, settled_end) if end else settled_end

    if effective_end < start:
        raise ValueError(
            f"no settled decision dates between {start} and {effective_end}: a "
            f"{holding_days}-trading-day hold needs roughly "
            f"{int(holding_days * _CALENDAR_PER_TRADING_DAY) + _SETTLE_SLACK_DAYS} "
            "calendar days to resolve, so recent dates have no realized return yet"
        )

    schedule = _decision_dates(start, effective_end, step_days)
    if not schedule:
        raise ValueError(f"no weekday decision dates in {start}..{effective_end}")

    points = tuple(
        EvalPoint(
            ticker=ticker,
            date=day,
            contaminated=bool(knowledge_cutoff and day <= knowledge_cutoff),
        )
        # Date-major ordering: a partial run that is interrupted still covers
        # the full universe for the dates it reached, rather than all dates for
        # only the first few tickers.
        for day in schedule
        for ticker in tickers
    )
    return EvaluationGrid(
        points=points, holding_days=holding_days, knowledge_cutoff=knowledge_cutoff
    )


def _decision_dates(start: str, end: str, step_days: int) -> list[str]:
    """Weekday decision dates from ``start`` to ``end`` every ``step_days``.

    Steps in calendar days, then snaps each result forward to the next weekday so
    a 7-day step does not land every point on the same weekend.
    """
    current = pd.Timestamp(start).normalize()
    limit = pd.Timestamp(end).normalize()
    out: list[str] = []
    while current <= limit:
        snapped = current
        while snapped.weekday() >= 5:  # Sat/Sun
            snapped += pd.Timedelta(days=1)
        if snapped > limit:
            break
        stamp = snapped.strftime("%Y-%m-%d")
        if not out or out[-1] != stamp:
            out.append(stamp)
        current += pd.Timedelta(days=step_days)
    return out
