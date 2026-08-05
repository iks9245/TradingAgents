"""Backtest orchestration: run strategies over a shared grid and score them.

The expensive strategy costs a full multi-agent run per evaluation point, so
every rating is appended to a JSONL cache as soon as it is produced. A run that
dies on point 137 of 200 — rate limit, expired key, laptop lid — resumes from
137 rather than re-spending on the first 136.

Nothing here decides whether the pipeline is any good; it produces the paired
numbers that let the reader decide. In particular the harness never drops a
strategy for underperforming, and reports contaminated (pre-knowledge-cutoff)
points separately rather than folding them into the headline.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from .grid import EvalPoint, EvaluationGrid
from .metrics import Decision, PairedComparison, StrategyMetrics, compare, summarize
from .prices import MissingPriceData, PriceCache
from .strategies import POSITION_MAPS, ShuffledRating, Strategy

logger = logging.getLogger(__name__)

DEFAULT_BENCHMARK = "SPY"


@dataclass
class BacktestResult:
    """Everything one backtest produced, in a form the reporter can render."""

    grid: EvaluationGrid
    decisions: dict[str, list[Decision]]
    metrics: dict[str, StrategyMetrics]
    comparisons: list[PairedComparison]
    benchmark: str
    position_map: str
    skipped: list[tuple[str, str]] = field(default_factory=list)
    clean_metrics: dict[str, StrategyMetrics] = field(default_factory=dict)
    clean_comparisons: list[PairedComparison] = field(default_factory=list)

    @property
    def strategy_names(self) -> list[str]:
        return list(self.decisions)


class DecisionCache:
    """Append-only JSONL store of (strategy, point) -> rating.

    Append-only rather than rewritten so a crash mid-write cannot corrupt
    already-earned decisions; a truncated final line is skipped on load.
    """

    def __init__(self, path: str | Path | None):
        self._path = Path(path).expanduser() if path else None
        self._entries: dict[tuple[str, str], str] = {}
        if self._path and self._path.exists():
            self._load()

    def _load(self) -> None:
        skipped = 0
        with open(self._path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    self._entries[(record["strategy"], record["key"])] = record["rating"]
                except (json.JSONDecodeError, KeyError, TypeError):
                    skipped += 1
        if skipped:
            logger.warning("Skipped %d unreadable line(s) in %s", skipped, self._path)
        logger.info("Loaded %d cached decision(s) from %s", len(self._entries), self._path)

    def get(self, strategy: str, point: EvalPoint) -> str | None:
        return self._entries.get((strategy, point.key))

    def put(self, strategy: str, point: EvalPoint, rating: str) -> None:
        self._entries[(strategy, point.key)] = rating
        if not self._path:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "strategy": strategy,
            "key": point.key,
            "ticker": point.ticker,
            "date": point.date,
            "rating": rating,
        }
        with open(self._path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def ratings_for(self, strategy: str) -> list[str]:
        return [rating for (name, _), rating in self._entries.items() if name == strategy]

    def __len__(self) -> int:
        return len(self._entries)


class Backtest:
    """Score a set of strategies against one another on a shared grid."""

    def __init__(
        self,
        grid: EvaluationGrid,
        *,
        benchmark: str = DEFAULT_BENCHMARK,
        position_map: str = "signed",
        entry_offset: int = 1,
        cache_path: str | Path | None = None,
        price_cache: PriceCache | None = None,
        bootstrap_iterations: int = 10_000,
        seed: int = 0,
    ):
        if position_map not in POSITION_MAPS:
            raise ValueError(
                f"unknown position_map {position_map!r}; expected one of {sorted(POSITION_MAPS)}"
            )
        self.grid = grid
        self.benchmark = benchmark
        self.position_map_name = position_map
        self._positions = POSITION_MAPS[position_map]
        self._entry_offset = entry_offset
        self._cache = DecisionCache(cache_path)
        self._prices = price_cache or PriceCache()
        self._bootstrap_iterations = bootstrap_iterations
        self._seed = seed
        self._outcomes: dict[str, object] = {}
        self._skipped: list[tuple[str, str]] = []

    def prepare(self) -> None:
        """Load prices and resolve the realized outcome for every grid point.

        Runs before any strategy so a point with no usable price data is dropped
        from the grid *for every strategy equally*, keeping the comparison paired.
        """
        self._prices.prefetch(
            [*self.grid.tickers, self.benchmark], start=self.grid.start, end=self.grid.end
        )
        for point in self.grid:
            try:
                self._outcomes[point.key] = self._prices.forward_return(
                    point.ticker,
                    point.date,
                    self.grid.holding_days,
                    benchmark=self.benchmark,
                    entry_offset=self._entry_offset,
                )
            except (MissingPriceData, ValueError) as exc:
                self._skipped.append((point.key, str(exc)))

        if self._skipped:
            logger.warning(
                "Dropped %d of %d grid point(s) with no realized return",
                len(self._skipped),
                len(self.grid),
            )
        if not self._outcomes:
            raise RuntimeError(
                "no grid point has a realized return; check the date range, the "
                "tickers, and network access to the price vendor"
            )

    def run_strategy(self, strategy: Strategy, *, progress=None) -> list[Decision]:
        """Rate every scorable point with ``strategy``, using and filling the cache."""
        if not self._outcomes:
            self.prepare()

        decisions: list[Decision] = []
        for point in self.grid:
            outcome = self._outcomes.get(point.key)
            if outcome is None:
                continue  # no realized return; dropped for all strategies alike

            rating = self._cache.get(strategy.name, point)
            if rating is None:
                try:
                    rating = strategy.decide(point)
                except Exception as exc:
                    # One failed point must not discard the decisions already
                    # paid for; it is recorded and the run continues.
                    logger.warning("%s failed on %s: %s", strategy.name, point.key, exc)
                    self._skipped.append((f"{strategy.name}:{point.key}", str(exc)))
                    continue
                self._cache.put(strategy.name, point, rating)

            position = self._positions.get(rating)
            if position is None:
                logger.warning(
                    "%s returned unrecognized rating %r on %s; scoring as Hold",
                    strategy.name, rating, point.key,
                )
                position = 0.0

            decisions.append(
                Decision(
                    strategy=strategy.name,
                    point=point,
                    rating=rating,
                    position=position,
                    outcome=outcome,
                )
            )
            if progress is not None:
                progress(strategy.name, point, rating)

        return decisions

    def run(
        self,
        strategy: Strategy,
        baselines: list[Strategy],
        *,
        include_matched_random: bool = True,
        progress=None,
    ) -> BacktestResult:
        """Run ``strategy`` and its ``baselines``, then score all of them.

        When ``include_matched_random`` is set, a :class:`ShuffledRating`
        baseline is built *after* the strategy runs, from that strategy's own
        realized rating distribution. It therefore cannot be constructed up
        front — it is the strategy's bullishness with the assignment randomized.
        """
        self.prepare()

        all_decisions: dict[str, list[Decision]] = {}
        strategy_decisions = self.run_strategy(strategy, progress=progress)
        if not strategy_decisions:
            raise RuntimeError(f"strategy {strategy.name!r} produced no scorable decisions")
        all_decisions[strategy.name] = strategy_decisions

        runnable = list(baselines)
        if include_matched_random:
            runnable.append(
                ShuffledRating.from_ratings(
                    [d.rating for d in strategy_decisions], seed=self._seed
                )
            )

        for baseline in runnable:
            decisions = self.run_strategy(baseline, progress=progress)
            if decisions:
                all_decisions[baseline.name] = decisions

        metrics = {name: summarize(items) for name, items in all_decisions.items()}
        comparisons = [
            compare(
                strategy_decisions,
                all_decisions[baseline.name],
                iterations=self._bootstrap_iterations,
                seed=self._seed,
            )
            for baseline in runnable
            if baseline.name in all_decisions
        ]

        clean_metrics, clean_comparisons = self._score_clean_subset(all_decisions, strategy, runnable)

        return BacktestResult(
            grid=self.grid,
            decisions=all_decisions,
            metrics=metrics,
            comparisons=comparisons,
            benchmark=self.benchmark,
            position_map=self.position_map_name,
            skipped=self._skipped,
            clean_metrics=clean_metrics,
            clean_comparisons=clean_comparisons,
        )

    def _score_clean_subset(self, all_decisions, strategy, baselines):
        """Re-score using only points after the knowledge cutoff.

        Reported alongside the full-sample numbers rather than instead of them:
        a large gap between the two is direct evidence that apparent skill came
        from the model remembering the outcome rather than forecasting it.
        """
        if not self.grid.knowledge_cutoff or self.grid.contaminated_count == 0:
            return {}, []

        clean = {
            name: [d for d in items if not d.point.contaminated]
            for name, items in all_decisions.items()
        }
        clean = {name: items for name, items in clean.items() if items}
        if strategy.name not in clean:
            logger.warning("No uncontaminated points remain; clean-subset metrics skipped")
            return {}, []

        metrics = {name: summarize(items) for name, items in clean.items()}
        comparisons = [
            compare(
                clean[strategy.name],
                clean[baseline.name],
                iterations=self._bootstrap_iterations,
                seed=self._seed,
            )
            for baseline in baselines
            if baseline.name in clean
        ]
        return metrics, comparisons
