"""Scoring and significance: alpha attribution, IC, and the clustered bootstrap."""

import pytest

from tradingagents.backtest.grid import EvalPoint
from tradingagents.backtest.metrics import Decision, compare, spearman, summarize
from tradingagents.backtest.prices import ForwardReturn


def _decision(strategy, ticker, date, position, raw, bench=0.0, rating="Buy"):
    return Decision(
        strategy=strategy,
        point=EvalPoint(ticker=ticker, date=date),
        rating=rating,
        position=position,
        outcome=ForwardReturn(
            ticker=ticker, decision_date=date, entry_date=date, exit_date=date,
            holding_days=21, raw_return=raw, benchmark_return=bench,
        ),
    )


@pytest.mark.unit
def test_alpha_pnl_removes_the_benchmark():
    """A long position in a market that rose exactly as much contributes nothing."""
    d = _decision("s", "AAPL", "2025-01-02", position=1.0, raw=0.05, bench=0.05)
    assert d.raw_pnl == pytest.approx(0.05)
    assert d.alpha_pnl == pytest.approx(0.0)


@pytest.mark.unit
def test_short_position_profits_from_a_decline():
    d = _decision("s", "AAPL", "2025-01-02", position=-1.0, raw=-0.10, rating="Sell")
    assert d.alpha_pnl == pytest.approx(0.10)


@pytest.mark.unit
def test_hit_rate_excludes_holds():
    """An all-Hold strategy takes no risk; it must not report a 100% hit rate."""
    decisions = [
        _decision("s", "A", "2025-01-02", 0.0, 0.05, rating="Hold"),
        _decision("s", "B", "2025-01-02", 0.0, -0.05, rating="Hold"),
        _decision("s", "C", "2025-01-02", 1.0, 0.05),
    ]
    assert summarize(decisions).hit_rate == pytest.approx(1.0)  # 1 of 1 active

    all_holds = [d for d in decisions if d.position == 0.0]
    assert summarize(all_holds).hit_rate == 0.0


@pytest.mark.unit
def test_information_coefficient_is_positive_when_ratings_rank_correctly():
    decisions = [
        _decision("s", "A", "2025-01-02", 1.0, 0.10),
        _decision("s", "B", "2025-01-02", 0.5, 0.05),
        _decision("s", "C", "2025-01-02", -0.5, -0.05),
        _decision("s", "D", "2025-01-02", -1.0, -0.10),
    ]
    assert summarize(decisions).information_coefficient == pytest.approx(1.0)


@pytest.mark.unit
def test_information_coefficient_is_zero_for_a_constant_strategy():
    """Always-Buy has no rank variation, so rank correlation is undefined -> 0."""
    decisions = [
        _decision("s", "A", "2025-01-02", 1.0, 0.10),
        _decision("s", "B", "2025-01-02", 1.0, -0.05),
    ]
    assert summarize(decisions).information_coefficient == 0.0


@pytest.mark.unit
def test_summarize_rejects_mixed_strategies():
    with pytest.raises(ValueError, match="multiple strategies"):
        summarize([
            _decision("a", "A", "2025-01-02", 1.0, 0.1),
            _decision("b", "B", "2025-01-02", 1.0, 0.1),
        ])


@pytest.mark.unit
def test_compare_detects_a_real_edge():
    """A strategy right on every point beats one wrong on every point."""
    dates = [f"2025-{m:02d}-03" for m in range(1, 13)]
    good, bad = [], []
    for i, date in enumerate(dates):
        for ticker in ("A", "B", "C"):
            raw = 0.05 if i % 2 == 0 else -0.05
            good.append(_decision("good", ticker, date, 1.0 if raw > 0 else -1.0, raw))
            bad.append(_decision("bad", ticker, date, -1.0 if raw > 0 else 1.0, raw))

    result = compare(good, bad, iterations=2000, seed=1)
    assert result.mean_difference > 0
    assert result.significant
    assert result.ci_low > 0


@pytest.mark.unit
def test_compare_finds_nothing_when_the_two_are_identical():
    decisions_a = [_decision("a", "A", f"2025-{m:02d}-03", 1.0, 0.01) for m in range(1, 13)]
    decisions_b = [_decision("b", "A", f"2025-{m:02d}-03", 1.0, 0.01) for m in range(1, 13)]

    result = compare(decisions_a, decisions_b, iterations=1000, seed=1)
    assert result.mean_difference == pytest.approx(0.0)
    assert result.p_value == 1.0
    assert not result.significant


@pytest.mark.unit
def test_bootstrap_clusters_by_date():
    """Same-day decisions share a market factor and are not independent.

    Twelve dates x 20 identical tickers carries no more information than 12
    dates x 1 ticker. A per-decision bootstrap would narrow the interval by
    sqrt(20); clustering by date must not.
    """
    def build(n_tickers):
        strategy, baseline = [], []
        for m in range(1, 13):
            date = f"2025-{m:02d}-03"
            raw = 0.05 if m % 2 else -0.03
            for t in range(n_tickers):
                strategy.append(_decision("s", f"T{t}", date, 1.0, raw))
                baseline.append(_decision("b", f"T{t}", date, 0.0, raw))
        return strategy, baseline

    narrow = compare(*build(1), iterations=2000, seed=7)
    wide = compare(*build(20), iterations=2000, seed=7)

    assert narrow.mean_difference == pytest.approx(wide.mean_difference)
    narrow_width = narrow.ci_high - narrow.ci_low
    wide_width = wide.ci_high - wide.ci_low
    # Duplicating tickers within each date adds no independent information, so
    # the interval must stay essentially the same width.
    assert wide_width == pytest.approx(narrow_width, rel=0.15)


@pytest.mark.unit
def test_compare_requires_shared_points():
    with pytest.raises(ValueError, match="no shared evaluation points"):
        compare(
            [_decision("a", "A", "2025-01-02", 1.0, 0.1)],
            [_decision("b", "B", "2025-02-03", 1.0, 0.1)],
        )


@pytest.mark.unit
def test_compare_drops_unpaired_points():
    strategy = [
        _decision("s", "A", "2025-01-02", 1.0, 0.10),
        _decision("s", "B", "2025-01-02", 1.0, 0.10),
    ]
    baseline = [_decision("b", "A", "2025-01-02", 0.0, 0.10)]
    assert compare(strategy, baseline, iterations=200).n == 1


@pytest.mark.unit
@pytest.mark.parametrize(
    ("xs", "ys", "expected"),
    [
        ([1, 2, 3], [1, 2, 3], 1.0),
        ([1, 2, 3], [3, 2, 1], -1.0),
        ([1, 1, 1], [1, 2, 3], 0.0),   # no rank variation
        ([1], [1], 0.0),               # too few points
    ],
)
def test_spearman_edge_cases(xs, ys, expected):
    assert spearman(xs, ys) == pytest.approx(expected)


@pytest.mark.unit
def test_spearman_handles_ties_with_average_ranks():
    assert spearman([1, 2, 2, 3], [1, 2, 2, 3]) == pytest.approx(1.0)


@pytest.mark.unit
def test_spearman_rejects_length_mismatch():
    with pytest.raises(ValueError, match="length mismatch"):
        spearman([1, 2], [1])
