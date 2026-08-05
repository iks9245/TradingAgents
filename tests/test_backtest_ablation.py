"""Ablation runner: arm naming, cache isolation, and the comparison matrix.

Offline throughout — arms are built by an injected ``strategy_factory`` so no
LLM client is ever constructed.
"""

import pandas as pd
import pytest

from tradingagents.backtest.ablation import (
    ALL_ANALYSTS,
    SUITES,
    AblationArm,
    analysts_drop_one,
    analysts_solo,
    debate_depth,
    dedupe_arms,
    estimated_runs,
    reference_arm,
    resolve_suite,
    risk_depth,
    run_ablation,
)
from tradingagents.backtest.ablation_cli import parse_arms
from tradingagents.backtest.ablation_report import (
    minimum_detectable_effect,
    render_ablation_report,
    render_ablation_summary,
)
from tradingagents.backtest.grid import build_grid
from tradingagents.backtest.harness import Backtest
from tradingagents.backtest.prices import PriceCache
from tradingagents.backtest.strategies import AlwaysRating

TODAY = pd.Timestamp("2026-06-01").date()


@pytest.fixture()
def grid():
    return build_grid(
        ["AAA", "BBB"], start="2025-01-06", end="2025-12-01",
        holding_days=5, step_days=14, today=TODAY,
    )


@pytest.fixture()
def prices():
    cache = PriceCache()
    index = pd.bdate_range(start="2024-12-01", periods=350)
    cache.load_frame("AAA", pd.DataFrame({"Close": [100 * 1.002 ** i for i in range(350)]}, index=index))
    cache.load_frame("BBB", pd.DataFrame({"Close": [100 * 0.998 ** i for i in range(350)]}, index=index))
    cache.load_frame("SPY", pd.DataFrame({"Close": [100.0] * 350}, index=index))
    return cache


class FixedArmStrategy:
    """Stands in for the LLM pipeline: each arm emits one fixed rating."""

    def __init__(self, name, rating):
        self.name = name
        self._rating = rating
        self.calls = 0

    def decide(self, point):
        self.calls += 1
        return self._rating


# --- arm naming and identity ------------------------------------------------


@pytest.mark.unit
def test_arm_name_encodes_the_analyst_selection():
    assert reference_arm().name == "ta[analysts=all]"
    assert AblationArm("m", selected_analysts=("market",)).name == "ta[analysts=market]"


@pytest.mark.unit
def test_arm_name_is_order_independent():
    """Two spellings of one configuration must share cached decisions, not duplicate them."""
    a = AblationArm("x", selected_analysts=("news", "market"))
    b = AblationArm("y", selected_analysts=("market", "news"))
    assert a.name == b.name == "ta[analysts=market+news]"


@pytest.mark.unit
def test_arm_name_encodes_config_overrides_deterministically():
    arm = AblationArm("d", config_overrides={"max_debate_rounds": 3, "max_risk_discuss_rounds": 2})
    other = AblationArm("d", config_overrides={"max_risk_discuss_rounds": 2, "max_debate_rounds": 3})
    assert arm.name == other.name
    assert "max_debate_rounds=3" in arm.name


@pytest.mark.unit
def test_differently_configured_arms_never_share_a_name():
    """The cache is keyed on name; a collision would serve one arm's decisions to another."""
    names = {arm.name for arm in analysts_drop_one() + analysts_solo() + debate_depth()}
    all_arms = analysts_drop_one() + analysts_solo() + debate_depth()
    assert len(names) == len(dedupe_arms(all_arms))


@pytest.mark.unit
def test_arm_rejects_unknown_or_empty_analysts():
    with pytest.raises(ValueError, match="unknown analysts"):
        AblationArm("bad", selected_analysts=("astrology",))
    with pytest.raises(ValueError, match="selects no analysts"):
        AblationArm("empty", selected_analysts=())


# --- suites -----------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("suite", sorted(SUITES))
def test_every_suite_builds_valid_arms(suite):
    arms = resolve_suite(suite)
    assert arms
    assert len({a.name for a in arms}) == len(dedupe_arms(arms))


@pytest.mark.unit
def test_drop_one_suite_removes_exactly_one_analyst_per_arm():
    arms = analysts_drop_one()
    assert arms[0] == reference_arm()
    ablated = arms[1:]
    assert len(ablated) == len(ALL_ANALYSTS)
    for arm in ablated:
        assert len(arm.selected_analysts) == len(ALL_ANALYSTS) - 1


@pytest.mark.unit
def test_solo_suite_runs_one_analyst_per_arm():
    for arm in analysts_solo()[1:]:
        assert len(arm.selected_analysts) == 1


@pytest.mark.unit
def test_depth_suites_vary_only_the_relevant_key():
    assert [a.config_overrides["max_debate_rounds"] for a in debate_depth((1, 2))] == [1, 2]
    assert [a.config_overrides["max_risk_discuss_rounds"] for a in risk_depth((1, 3))] == [1, 3]


@pytest.mark.unit
def test_unknown_suite_is_rejected():
    with pytest.raises(ValueError, match="unknown suite"):
        resolve_suite("vibes")


@pytest.mark.unit
def test_duplicate_arms_are_dropped_before_spending():
    """debate_depth includes the shipped default, so suites can legitimately collide."""
    arms = [reference_arm(), reference_arm(), AblationArm("m", selected_analysts=("market",))]
    assert len(dedupe_arms(arms)) == 2


@pytest.mark.unit
def test_estimated_runs_counts_unique_arms_times_points(grid):
    arms = [reference_arm(), reference_arm(), AblationArm("m", selected_analysts=("market",))]
    assert estimated_runs(grid, arms) == 2 * len(grid)


# --- CLI arm parsing --------------------------------------------------------


@pytest.mark.unit
def test_parse_arms_handles_all_and_combinations():
    arms = parse_arms("all,market,market+news")
    assert [a.name for a in arms] == [
        "ta[analysts=all]",
        "ta[analysts=market]",
        "ta[analysts=market+news]",
    ]


@pytest.mark.unit
def test_parse_arms_rejects_empty_spec():
    with pytest.raises(ValueError, match="no arms parsed"):
        parse_arms("  ,  ")


# --- running ----------------------------------------------------------------


def _run(grid, prices, arms, ratings, **kwargs):
    harness = Backtest(grid, price_cache=prices, bootstrap_iterations=400)
    return run_ablation(
        grid, arms, backtest=harness,
        strategy_factory=lambda arm: FixedArmStrategy(arm.name, ratings[arm.name]),
        **kwargs,
    )


@pytest.mark.unit
def test_every_arm_is_scored_on_identical_points(grid, prices):
    arms = [reference_arm(), AblationArm("m", selected_analysts=("market",))]
    result = _run(grid, prices, arms, {a.name: "Buy" for a in arms})

    scored = [{d.point.key for d in items} for items in result.decisions.values()]
    assert len({frozenset(s) for s in scored}) == 1


@pytest.mark.unit
def test_comparison_is_produced_per_non_reference_arm(grid, prices):
    arms = [
        reference_arm(),
        AblationArm("m", selected_analysts=("market",)),
        AblationArm("n", selected_analysts=("news",)),
    ]
    result = _run(grid, prices, arms, {a.name: "Buy" for a in arms})

    assert len(result.vs_reference) == 2
    assert all(c.baseline == result.reference.name for c in result.vs_reference)
    assert len(result.vs_baseline) == 3  # every arm, reference included


@pytest.mark.unit
def test_identical_arms_show_no_difference_from_the_reference(grid, prices):
    arms = [reference_arm(), AblationArm("m", selected_analysts=("market",))]
    result = _run(grid, prices, arms, {a.name: "Buy" for a in arms})

    assert result.vs_reference[0].mean_difference == pytest.approx(0.0)
    assert not result.vs_reference[0].significant


@pytest.mark.unit
def test_a_genuinely_worse_arm_is_detected(grid, prices):
    """Positive control: the matrix must be able to find a difference, not just miss them."""
    ref = reference_arm()
    bad = AblationArm("inverted", selected_analysts=("market",))
    # AAA rises and BBB falls, so all-Buy beats all-Sell by a wide margin.
    result = _run(grid, prices, [ref, bad], {ref.name: "Buy", bad.name: "Sell"})

    diff = result.vs_reference[0]
    assert diff.mean_difference < 0
    assert diff.significant
    assert "worse" in render_ablation_report(result)


@pytest.mark.unit
def test_reference_defaults_to_the_first_arm(grid, prices):
    arms = [AblationArm("m", selected_analysts=("market",)), reference_arm()]
    result = _run(grid, prices, arms, {a.name: "Buy" for a in arms})
    assert result.reference.name == "ta[analysts=market]"


@pytest.mark.unit
def test_explicit_reference_is_added_when_absent(grid, prices):
    ref = reference_arm()
    arms = [AblationArm("m", selected_analysts=("market",))]
    result = _run(
        grid, prices, arms, {ref.name: "Buy", "ta[analysts=market]": "Buy"}, reference=ref
    )
    assert result.reference.name == ref.name
    assert ref.name in result.decisions


@pytest.mark.unit
def test_arms_do_not_share_cached_decisions(tmp_path, grid, prices):
    """Two arms with different configs must each pay for their own decisions."""
    cache_path = tmp_path / "ablation.jsonl"
    ref = reference_arm()
    solo = AblationArm("m", selected_analysts=("market",))
    built = {}

    def factory(arm):
        strategy = FixedArmStrategy(arm.name, "Buy" if arm.name == ref.name else "Sell")
        built[arm.name] = strategy
        return strategy

    harness = Backtest(grid, price_cache=prices, cache_path=cache_path, bootstrap_iterations=200)
    result = run_ablation(grid, [ref, solo], backtest=harness, strategy_factory=factory)

    assert built[ref.name].calls > 0 and built[solo.name].calls > 0
    assert {d.rating for d in result.decisions[ref.name]} == {"Buy"}
    assert {d.rating for d in result.decisions[solo.name]} == {"Sell"}


@pytest.mark.unit
def test_second_ablation_run_reuses_the_shared_cache(tmp_path, grid, prices):
    cache_path = tmp_path / "ablation.jsonl"
    arms = [reference_arm(), AblationArm("m", selected_analysts=("market",))]
    ratings = {a.name: "Buy" for a in arms}

    for _ in range(1):
        harness = Backtest(grid, price_cache=prices, cache_path=cache_path, bootstrap_iterations=200)
        run_ablation(
            grid, arms, backtest=harness,
            strategy_factory=lambda arm: FixedArmStrategy(arm.name, ratings[arm.name]),
        )

    built = {}

    def factory(arm):
        built[arm.name] = FixedArmStrategy(arm.name, ratings[arm.name])
        return built[arm.name]

    harness = Backtest(grid, price_cache=prices, cache_path=cache_path, bootstrap_iterations=200)
    run_ablation(grid, arms, backtest=harness, strategy_factory=factory)

    assert all(s.calls == 0 for s in built.values()), "cached arms must not re-decide"


@pytest.mark.unit
def test_run_ablation_requires_arms(grid, prices):
    with pytest.raises(ValueError, match="at least one arm"):
        _run(grid, prices, [], {})


# --- reporting --------------------------------------------------------------


@pytest.mark.unit
def test_report_renders_with_verdict_and_caveats(grid, prices):
    arms = [reference_arm(), AblationArm("m", selected_analysts=("market",))]
    result = _run(grid, prices, arms, {a.name: "Buy" for a in arms})

    report = render_ablation_report(result)
    assert "# Ablation Report" in report
    assert "## Arm minus reference" in report
    assert "**Verdict**" in report
    # The asymmetry that makes a null result actionable must be stated.
    assert "not paying for itself" in report
    assert "Only analyst arms are structural" in report
    assert render_ablation_summary(result)


@pytest.mark.unit
def test_null_verdict_says_the_removed_work_did_not_pay(grid, prices):
    arms = [reference_arm(), AblationArm("m", selected_analysts=("market",))]
    result = _run(grid, prices, arms, {a.name: "Buy" for a in arms})
    assert "no arm differs measurably" in render_ablation_report(result)


@pytest.mark.unit
def test_report_states_the_resolvable_effect_size(grid, prices):
    """An underpowered null and a true null look identical without this."""
    arms = [reference_arm(), AblationArm("m", selected_analysts=("market",))]
    result = _run(grid, prices, arms, {a.name: "Buy" for a in arms})
    assert "Resolvable effect" in render_ablation_report(result)


@pytest.mark.unit
def test_minimum_detectable_effect_is_the_interval_half_width():
    from tradingagents.backtest.metrics import PairedComparison

    def comparison(low, high):
        return PairedComparison(
            strategy="s", baseline="b", n=10, mean_difference=(low + high) / 2,
            ci_low=low, ci_high=high, p_value=0.5, confidence=0.95,
        )

    assert minimum_detectable_effect([comparison(-0.01, 0.01)]) == pytest.approx(0.01)
    # Median across arms, so one wide arm does not set the headline.
    assert minimum_detectable_effect(
        [comparison(-0.01, 0.01), comparison(-0.02, 0.02), comparison(-0.09, 0.09)]
    ) == pytest.approx(0.02)
    assert minimum_detectable_effect([]) == 0.0


@pytest.mark.unit
def test_report_uses_human_labels_not_cache_names(grid, prices):
    arms = [reference_arm(), AblationArm("without news", selected_analysts=("market", "social"))]
    result = _run(grid, prices, arms, {a.name: "Buy" for a in arms})
    assert "without news" in render_ablation_report(result)


@pytest.mark.unit
def test_baseline_defaults_to_buy_and_hold(grid, prices):
    arms = [reference_arm()]
    result = _run(grid, prices, arms, {a.name: "Buy" for a in arms})
    assert AlwaysRating("Buy").name in result.metrics
