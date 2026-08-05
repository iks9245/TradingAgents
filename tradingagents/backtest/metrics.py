"""Scoring and significance testing for backtest results.

Two design choices here matter more than the formulas:

**Alpha is the headline, not raw return.** Over any rising window a long-biased
strategy shows a positive raw return while adding nothing. Subtracting the
benchmark over the identical bar window removes that, leaving what the pipeline
actually contributed.

**The bootstrap clusters by date.** Twelve tickers rated on the same day are not
twelve independent observations — they share a market factor, and on that day
they tend to be right or wrong together. Resampling individual decisions would
treat them as independent and produce confidence intervals far too narrow, which
is the standard way a backtest manufactures significance that does not survive
contact with live trading. Resampling whole dates (with all their tickers) keeps
that correlation intact.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .grid import EvalPoint
from .prices import ForwardReturn


@dataclass(frozen=True)
class Decision:
    """One strategy's rating for one point, joined to the realized outcome."""

    strategy: str
    point: EvalPoint
    rating: str
    position: float
    outcome: ForwardReturn

    @property
    def raw_pnl(self) -> float:
        """Return contribution before removing the benchmark."""
        return self.position * self.outcome.raw_return

    @property
    def alpha_pnl(self) -> float:
        """Return contribution net of the benchmark — the headline number."""
        return self.position * self.outcome.alpha_return


@dataclass(frozen=True)
class StrategyMetrics:
    """Summary statistics for one strategy over one grid."""

    strategy: str
    n: int
    n_dates: int
    mean_raw: float
    mean_alpha: float
    std_alpha: float
    t_stat: float
    hit_rate: float
    information_coefficient: float
    rating_counts: dict[str, int] = field(default_factory=dict)

    @property
    def mean_alpha_pct(self) -> float:
        return self.mean_alpha * 100.0


@dataclass(frozen=True)
class PairedComparison:
    """Strategy minus baseline, measured on the identical grid."""

    strategy: str
    baseline: str
    n: int
    mean_difference: float
    ci_low: float
    ci_high: float
    p_value: float
    confidence: float

    @property
    def significant(self) -> bool:
        """True when the confidence interval excludes zero."""
        return self.ci_low > 0.0 or self.ci_high < 0.0


def summarize(decisions: list[Decision]) -> StrategyMetrics:
    """Compute summary statistics for one strategy's decisions."""
    if not decisions:
        raise ValueError("cannot summarize an empty decision list")

    names = {d.strategy for d in decisions}
    if len(names) > 1:
        raise ValueError(f"decisions span multiple strategies: {sorted(names)}")

    alphas = [d.alpha_pnl for d in decisions]
    raws = [d.raw_pnl for d in decisions]
    n = len(decisions)
    mean_alpha = _mean(alphas)
    std_alpha = _stdev(alphas)

    # Hit rate counts only decisions that took a position: a Hold contributes
    # exactly zero and would otherwise pad the denominator (or, if counted as a
    # win, let an all-Hold strategy report a 100% hit rate).
    active = [d for d in decisions if d.position != 0.0]
    hits = sum(1 for d in active if d.alpha_pnl > 0)
    hit_rate = hits / len(active) if active else 0.0

    return StrategyMetrics(
        strategy=next(iter(names)),
        n=n,
        n_dates=len({d.point.date for d in decisions}),
        mean_raw=_mean(raws),
        mean_alpha=mean_alpha,
        std_alpha=std_alpha,
        # Naive t-stat, reported for familiarity only. It assumes independent
        # observations, which the grid violates; trust the clustered bootstrap
        # interval instead.
        t_stat=(mean_alpha / (std_alpha / math.sqrt(n))) if std_alpha > 0 and n > 1 else 0.0,
        hit_rate=hit_rate,
        information_coefficient=spearman(
            [d.position for d in decisions], [d.outcome.alpha_return for d in decisions]
        ),
        rating_counts=dict(Counter(d.rating for d in decisions)),
    )


def compare(
    strategy_decisions: list[Decision],
    baseline_decisions: list[Decision],
    *,
    iterations: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> PairedComparison:
    """Paired, date-clustered bootstrap of strategy-minus-baseline alpha.

    Both lists must cover the same evaluation points; points present in only one
    are dropped, since an unpaired difference is not attributable to the strategy.
    """
    by_key_strategy = {d.point.key: d for d in strategy_decisions}
    by_key_baseline = {d.point.key: d for d in baseline_decisions}
    shared = sorted(set(by_key_strategy) & set(by_key_baseline))
    if not shared:
        raise ValueError("no shared evaluation points between the two strategies")

    # Group paired differences by decision date — the resampling unit.
    clusters: dict[str, list[float]] = defaultdict(list)
    for key in shared:
        diff = by_key_strategy[key].alpha_pnl - by_key_baseline[key].alpha_pnl
        clusters[by_key_strategy[key].point.date].append(diff)

    cluster_values = list(clusters.values())
    observed = _mean([d for values in cluster_values for d in values])

    means = _bootstrap_cluster_means(cluster_values, iterations=iterations, seed=seed)
    tail = (1.0 - confidence) / 2.0
    ci_low = _quantile(means, tail)
    ci_high = _quantile(means, 1.0 - tail)

    p_value = _bootstrap_p_value(means, observed)

    return PairedComparison(
        strategy=strategy_decisions[0].strategy,
        baseline=baseline_decisions[0].strategy,
        n=len(shared),
        mean_difference=observed,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        confidence=confidence,
    )


def _bootstrap_p_value(means: list[float], observed: float) -> float:
    """Two-sided bootstrap p-value for ``observed`` being non-zero.

    Counts resamples falling on the opposite side of zero from the observed
    effect and doubles that fraction. The result is floored at ``1/len(means)``:
    zero crossings in a finite bootstrap means "smaller than this resolution",
    not zero. An observed effect of exactly zero yields 1.0.
    """
    if not means or observed == 0.0:
        return 1.0
    crossings = sum(1 for m in means if (m <= 0.0) == (observed > 0.0))
    return min(1.0, max(1.0 / len(means), 2.0 * crossings / len(means)))


def _bootstrap_cluster_means(
    clusters: list[list[float]], *, iterations: int, seed: int
) -> list[float]:
    """Means of ``iterations`` resamples drawn over whole clusters."""
    rng = random.Random(seed)
    n_clusters = len(clusters)
    means: list[float] = []
    for _ in range(iterations):
        total = 0.0
        count = 0
        for _ in range(n_clusters):
            picked = clusters[rng.randrange(n_clusters)]
            total += sum(picked)
            count += len(picked)
        if count:
            means.append(total / count)
    return sorted(means)


def spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation, tie-aware.

    Implemented here rather than pulled from scipy, which is not a dependency.
    Returns 0.0 when either input has no variation in rank (e.g. a constant
    strategy), where the correlation is undefined.
    """
    if len(xs) != len(ys):
        raise ValueError(f"length mismatch: {len(xs)} vs {len(ys)}")
    if len(xs) < 2:
        return 0.0
    return _pearson(_ranks(xs), _ranks(ys))


def _ranks(values: list[float]) -> list[float]:
    """Average ranks (1-based), ties sharing the mean of their positions."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float:
    mean_x, mean_y = _mean(xs), _mean(ys)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x <= 0 or var_y <= 0:
        return 0.0
    return cov / math.sqrt(var_x * var_y)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _stdev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def _quantile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolated quantile of an already-sorted list."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return sorted_values[int(pos)]
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (pos - low)
