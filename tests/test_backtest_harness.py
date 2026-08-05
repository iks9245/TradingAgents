"""Harness orchestration: cache resumption, paired dropping, baseline behaviour.

No LLM and no network: strategies are cheap stand-ins and prices are installed
via ``PriceCache.load_frame``.
"""

import json

import pandas as pd
import pytest

from tradingagents.backtest.grid import EvalPoint, build_grid
from tradingagents.backtest.harness import Backtest, DecisionCache
from tradingagents.backtest.prices import PriceCache
from tradingagents.backtest.report import render_report, render_summary
from tradingagents.backtest.strategies import (
    LONG_ONLY_POSITIONS,
    SIGNED_POSITIONS,
    AlwaysRating,
    ShuffledRating,
    UniformRandomRating,
)

TODAY = pd.Timestamp("2026-06-01").date()


class CountingStrategy:
    """Records every point it was asked about, so cache hits are observable."""

    def __init__(self, rating="Buy", name="counting"):
        self.name = name
        self._rating = rating
        self.calls = []

    def decide(self, point):
        self.calls.append(point.key)
        return self._rating


class FlakyStrategy:
    """Fails on one specific point; every other point must still be scored."""

    name = "flaky"

    def __init__(self, fail_on):
        self._fail_on = fail_on

    def decide(self, point):
        if point.key == self._fail_on:
            raise RuntimeError("simulated provider outage")
        return "Buy"


@pytest.fixture()
def grid():
    return build_grid(
        ["AAA", "BBB"], start="2025-01-06", end="2025-04-01",
        holding_days=5, step_days=21, today=TODAY,
    )


@pytest.fixture()
def prices():
    cache = PriceCache()
    index = pd.bdate_range(start="2024-12-01", periods=200)
    cache.load_frame("AAA", pd.DataFrame({"Close": [100 * 1.001 ** i for i in range(200)]}, index=index))
    cache.load_frame("BBB", pd.DataFrame({"Close": [100 * 0.999 ** i for i in range(200)]}, index=index))
    cache.load_frame("SPY", pd.DataFrame({"Close": [100.0] * 200}, index=index))
    return cache


def _backtest(grid, prices, **kwargs):
    return Backtest(grid, price_cache=prices, bootstrap_iterations=200, **kwargs)


@pytest.mark.unit
def test_every_grid_point_is_scored(grid, prices):
    decisions = _backtest(grid, prices).run_strategy(AlwaysRating("Buy"))
    assert len(decisions) == len(grid)


@pytest.mark.unit
def test_cached_decisions_are_not_recomputed(tmp_path, grid, prices):
    """The expensive path costs a full multi-agent run per point; resuming must skip it."""
    cache_path = tmp_path / "decisions.jsonl"

    first = CountingStrategy()
    _backtest(grid, prices, cache_path=cache_path).run_strategy(first)
    assert len(first.calls) == len(grid)

    second = CountingStrategy()
    decisions = _backtest(grid, prices, cache_path=cache_path).run_strategy(second)
    assert second.calls == []          # every point served from cache
    assert len(decisions) == len(grid)  # but still fully scored


@pytest.mark.unit
def test_partial_cache_resumes_only_the_missing_points(tmp_path, grid, prices):
    cache_path = tmp_path / "decisions.jsonl"
    cache = DecisionCache(cache_path)
    cache.put("counting", grid.points[0], "Sell")

    strategy = CountingStrategy()
    decisions = _backtest(grid, prices, cache_path=cache_path).run_strategy(strategy)

    assert grid.points[0].key not in strategy.calls
    assert len(strategy.calls) == len(grid) - 1
    assert decisions[0].rating == "Sell"       # cached value wins
    assert decisions[0].position == -1.0


@pytest.mark.unit
def test_corrupt_cache_line_is_skipped_not_fatal(tmp_path, grid, prices):
    """A crash mid-write must not make already-earned decisions unreadable."""
    cache_path = tmp_path / "decisions.jsonl"
    good = {"strategy": "counting", "key": grid.points[0].key, "rating": "Sell"}
    cache_path.write_text(json.dumps(good) + "\n" + '{"strategy": "counting", "key"', encoding="utf-8")

    loaded = DecisionCache(cache_path)
    assert loaded.get("counting", grid.points[0]) == "Sell"
    assert len(loaded) == 1


@pytest.mark.unit
def test_failed_point_is_recorded_and_the_run_continues(grid, prices):
    backtest = _backtest(grid, prices)
    decisions = backtest.run_strategy(FlakyStrategy(fail_on=grid.points[0].key))

    assert len(decisions) == len(grid) - 1
    assert all(d.point.key != grid.points[0].key for d in decisions)


@pytest.mark.unit
def test_points_without_prices_are_dropped_for_every_strategy_alike(prices):
    """Unpaired drops would break the comparison, so they happen before any strategy runs."""
    grid = build_grid(
        ["AAA", "MISSING"], start="2025-01-06", end="2025-03-01",
        holding_days=5, step_days=21, today=TODAY,
    )
    backtest = _backtest(grid, prices)
    result = backtest.run(AlwaysRating("Buy"), [UniformRandomRating(seed=1)])

    scored = {name: {d.point.key for d in items} for name, items in result.decisions.items()}
    assert len({frozenset(keys) for keys in scored.values()}) == 1
    assert all(not key.startswith("MISSING") for keys in scored.values() for key in keys)
    assert result.skipped


@pytest.mark.unit
def test_run_produces_a_comparison_per_baseline(grid, prices):
    result = _backtest(grid, prices).run(
        AlwaysRating("Buy", name="strategy_under_test"),
        [AlwaysRating("Hold", name="always_hold"), UniformRandomRating(seed=3)],
    )
    baselines = {c.baseline for c in result.comparisons}
    assert baselines == {"always_hold", "random_uniform", "random_matched"}
    assert all(c.strategy == "strategy_under_test" for c in result.comparisons)


@pytest.mark.unit
def test_matched_random_mirrors_the_strategy_rating_mix(grid, prices):
    """The strict baseline: same bullish tilt, randomized assignment."""
    result = _backtest(grid, prices).run(AlwaysRating("Buy", name="sut"), [])

    matched = result.metrics["random_matched"]
    # Strategy was all-Buy, so its shuffled twin can only ever draw Buy.
    assert matched.rating_counts == {"Buy": matched.n}
    # Which makes the two identical — exactly zero measured edge, as it should be.
    assert result.comparisons[0].mean_difference == pytest.approx(0.0)


@pytest.mark.unit
def test_contaminated_points_are_reported_separately(prices):
    grid = build_grid(
        ["AAA", "BBB"], start="2025-01-06", end="2025-05-01",
        holding_days=5, step_days=21, knowledge_cutoff="2025-03-01", today=TODAY,
    )
    result = _backtest(grid, prices).run(AlwaysRating("Buy", name="sut"), [UniformRandomRating()])

    assert result.clean_metrics, "expected an out-of-sample subset"
    assert result.clean_metrics["sut"].n < result.metrics["sut"].n
    assert result.clean_comparisons


@pytest.mark.unit
def test_no_cutoff_means_no_clean_subset(grid, prices):
    result = _backtest(grid, prices).run(AlwaysRating("Buy", name="sut"), [])
    assert result.clean_metrics == {}


@pytest.mark.unit
def test_random_baselines_are_deterministic_per_point():
    """Same seed, same point, same rating — regardless of visit order or resumption.

    Keyed on the point rather than drawn from a running stream, so a run that is
    interrupted and resumed reproduces the same baseline it would have produced
    in one pass.
    """
    points = [EvalPoint(ticker=f"T{i}", date="2025-01-06") for i in range(20)]

    forward = [UniformRandomRating(seed=5).decide(p) for p in points]
    reversed_order = [UniformRandomRating(seed=5).decide(p) for p in reversed(points)]
    assert forward == list(reversed(reversed_order))

    # A different seed must actually change the draw, or the seed is being ignored.
    assert forward != [UniformRandomRating(seed=6).decide(p) for p in points]


@pytest.mark.unit
def test_uniform_random_covers_the_whole_scale():
    points = [EvalPoint(ticker=f"T{i}", date="2025-01-06") for i in range(300)]
    ratings = {UniformRandomRating(seed=0).decide(p) for p in points}
    assert len(ratings) == 5


@pytest.mark.unit
def test_shuffled_rating_rejects_bad_input():
    with pytest.raises(ValueError, match="non-empty"):
        ShuffledRating([])
    with pytest.raises(ValueError, match="unknown ratings"):
        ShuffledRating(["Buy", "Moon"])


@pytest.mark.unit
def test_position_maps_differ_only_on_the_bearish_half():
    assert SIGNED_POSITIONS["Sell"] == -1.0
    assert LONG_ONLY_POSITIONS["Sell"] == 0.0
    assert SIGNED_POSITIONS["Buy"] == LONG_ONLY_POSITIONS["Buy"] == 1.0


@pytest.mark.unit
def test_long_only_map_never_shorts(grid, prices):
    result = _backtest(grid, prices, position_map="long_only").run(
        AlwaysRating("Sell", name="bear"), []
    )
    assert all(d.position >= 0 for d in result.decisions["bear"])


@pytest.mark.unit
def test_unknown_position_map_is_rejected(grid, prices):
    with pytest.raises(ValueError, match="unknown position_map"):
        _backtest(grid, prices, position_map="leveraged")


@pytest.mark.unit
def test_oracle_strategy_is_detected_as_significant(prices):
    """Positive control: a harness that can only ever say "no edge" is worthless.

    The oracle cheats — it rates by the sign of the realized return. If the
    pipeline is ever measured as having no edge, this test is what establishes
    that the measurement was capable of finding one.
    """
    grid = build_grid(
        ["AAA", "BBB"], start="2025-01-06", end="2025-12-01",
        holding_days=5, step_days=14, today=TODAY,
    )

    class Oracle:
        name = "oracle"

        def __init__(self, backtest):
            self._backtest = backtest

        def decide(self, point):
            outcome = self._backtest._outcomes[point.key]
            return "Buy" if outcome.alpha_return > 0 else "Sell"

    backtest = _backtest(grid, prices)
    backtest.prepare()
    result = backtest.run(Oracle(backtest), [AlwaysRating("Buy"), UniformRandomRating(seed=2)])

    for comparison in result.comparisons:
        assert comparison.mean_difference > 0, comparison.baseline
        assert comparison.significant, f"{comparison.baseline}: CI {comparison.ci_low}..{comparison.ci_high}"
    assert result.metrics["oracle"].hit_rate == pytest.approx(1.0)
    # The oracle only emits Buy/Sell, so its positions are two heavily tied rank
    # groups against continuous returns — that caps Spearman near 0.87 even with
    # a perfect sign call. A perfect IC of 1.0 needs the full five-tier spread.
    assert result.metrics["oracle"].information_coefficient > 0.8
    assert "beats every baseline" in render_report(result)


@pytest.mark.unit
def test_report_renders_and_states_a_verdict(grid, prices):
    result = _backtest(grid, prices).run(AlwaysRating("Buy", name="sut"), [UniformRandomRating()])

    report = render_report(result)
    assert "# Backtest Report" in report
    assert "**Verdict**" in report
    assert "random_matched" in report
    # The caveat about un-cutoff'd runs must appear when none was supplied.
    assert "No knowledge cutoff was supplied" in report
    assert render_summary(result)


@pytest.mark.unit
def test_report_warns_when_the_edge_is_not_significant(grid, prices):
    result = _backtest(grid, prices).run(AlwaysRating("Buy", name="sut"), [])
    assert "consistent with the strategy having no edge" in render_report(result)
