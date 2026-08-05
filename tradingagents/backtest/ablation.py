"""Ablation runner: price what each part of the pipeline actually contributes.

The graph has twelve decision nodes but only four of them fetch new information;
the other eight re-process the same analyst reports. Whether those eight change
the outcome is an empirical question, and this module answers it by running
configurations that differ in exactly one respect over an identical grid.

Every arm is compared against the **reference arm** (the full pipeline) with the
same paired, date-clustered bootstrap the baseline comparison uses. An arm whose
difference from the reference has a confidence interval spanning zero did not
measurably change the result — which, for an arm that removes work, means the
removed work was not earning its cost on this grid.

What can and cannot be ablated from config:

- **Analysts** genuinely come out of the graph. ``selected_analysts`` removes
  their nodes and their tools, so ``analysts_solo`` and ``analysts_drop_one``
  measure real structural changes.
- **Debate and risk rounds** only vary in *depth*. The graph hard-codes at least
  one Bull/Bear exchange and one pass through the three risk analysts, so
  ``debate_depth`` measures the marginal value of *more* debate, not the value
  of debate versus none. Removing those stages entirely needs a graph change,
  not a config flag.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .grid import EvaluationGrid
from .harness import Backtest
from .metrics import Decision, PairedComparison, StrategyMetrics, compare, summarize
from .strategies import AlwaysRating, Strategy, TradingAgentsStrategy, build_backtest_config

logger = logging.getLogger(__name__)

ALL_ANALYSTS = ("market", "social", "news", "fundamentals")


@dataclass(frozen=True)
class AblationArm:
    """One configuration of the pipeline under test.

    ``name`` is derived from the configuration rather than supplied, so two arms
    can never collide in the shared decision cache. That matters more than it
    looks: the cache is keyed on (strategy name, point), so a hand-written name
    reused across two different configurations would silently serve one arm's
    expensive decisions to the other and report them as identical.
    """

    label: str
    selected_analysts: tuple[str, ...] | None = None
    config_overrides: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.selected_analysts is not None:
            unknown = set(self.selected_analysts) - set(ALL_ANALYSTS)
            if unknown:
                raise ValueError(f"unknown analysts {sorted(unknown)}; expected {ALL_ANALYSTS}")
            if not self.selected_analysts:
                raise ValueError(f"arm {self.label!r} selects no analysts")

    @property
    def name(self) -> str:
        """Cache-safe identifier encoding everything that changes behaviour."""
        parts = []
        if self.selected_analysts is None:
            parts.append("analysts=all")
        else:
            # Canonical order, so ("news", "market") and ("market", "news") are
            # recognised as the same arm and share cached decisions.
            ordered = [a for a in ALL_ANALYSTS if a in self.selected_analysts]
            parts.append("analysts=" + "+".join(ordered))
        parts += [f"{key}={self.config_overrides[key]}" for key in sorted(self.config_overrides)]
        return "ta[" + ",".join(parts) + "]"

    def build(self, base_config: dict | None = None) -> TradingAgentsStrategy:
        """Construct the strategy this arm describes."""
        return TradingAgentsStrategy(
            config=build_backtest_config(base_config, **self.config_overrides),
            selected_analysts=self.selected_analysts,
            name=self.name,
        )


def reference_arm() -> AblationArm:
    """The full pipeline, as shipped. Every other arm is measured against it."""
    return AblationArm(label="full pipeline")


def analysts_drop_one() -> list[AblationArm]:
    """Full pipeline, plus one arm per analyst removed.

    The direct read on what each analyst contributes: if dropping the news
    analyst does not move the result, its tool calls are not paying for
    themselves.
    """
    arms = [reference_arm()]
    for analyst in ALL_ANALYSTS:
        remaining = tuple(a for a in ALL_ANALYSTS if a != analyst)
        arms.append(AblationArm(label=f"without {analyst}", selected_analysts=remaining))
    return arms


def analysts_solo() -> list[AblationArm]:
    """Full pipeline, plus one arm per analyst run alone.

    The complement of ``analysts_drop_one``: how much of the full pipeline's
    result one analyst reproduces on its own. Cheap arms, too — a solo run is a
    fraction of a full run's tokens.
    """
    arms = [reference_arm()]
    for analyst in ALL_ANALYSTS:
        arms.append(AblationArm(label=f"{analyst} only", selected_analysts=(analyst,)))
    return arms


def debate_depth(levels=(1, 2, 3)) -> list[AblationArm]:
    """Vary Bull/Bear debate rounds.

    Measures the marginal value of deeper debate, not of debate at all — the
    graph always runs at least one exchange. Note the cost: these arms add LLM
    calls rather than removing them.
    """
    return [
        AblationArm(label=f"debate x{level}", config_overrides={"max_debate_rounds": level})
        for level in levels
    ]


def risk_depth(levels=(1, 2, 3)) -> list[AblationArm]:
    """Vary rounds through the three risk analysts. Same caveat as debate depth."""
    return [
        AblationArm(label=f"risk x{level}", config_overrides={"max_risk_discuss_rounds": level})
        for level in levels
    ]


SUITES = {
    "analysts_drop_one": analysts_drop_one,
    "analysts_solo": analysts_solo,
    "debate_depth": debate_depth,
    "risk_depth": risk_depth,
}

DEFAULT_SUITE = "analysts_drop_one"


def resolve_suite(name: str) -> list[AblationArm]:
    if name not in SUITES:
        raise ValueError(f"unknown suite {name!r}; expected one of {', '.join(sorted(SUITES))}")
    return SUITES[name]()


def dedupe_arms(arms) -> list[AblationArm]:
    """Drop arms whose configuration is identical to an earlier one.

    ``debate_depth`` includes the shipped default among its levels, so a suite
    can legitimately contain two arms that are the same configuration under
    different labels. Running both would double that arm's cost for no
    information.
    """
    seen: dict[str, AblationArm] = {}
    for arm in arms:
        if arm.name in seen:
            logger.info("Skipping duplicate arm %r (same config as %r)", arm.label, seen[arm.name].label)
            continue
        seen[arm.name] = arm
    return list(seen.values())


@dataclass
class AblationResult:
    """Per-arm metrics plus each arm's paired difference from the reference."""

    grid: EvaluationGrid
    arms: list[AblationArm]
    reference: AblationArm
    metrics: dict[str, StrategyMetrics]
    vs_reference: list[PairedComparison]
    vs_baseline: list[PairedComparison]
    decisions: dict[str, list[Decision]] = field(default_factory=dict)
    skipped: list[tuple[str, str]] = field(default_factory=list)

    def label_for(self, name: str) -> str:
        for arm in self.arms:
            if arm.name == name:
                return arm.label
        return name


def run_ablation(
    grid: EvaluationGrid,
    arms: list[AblationArm],
    *,
    backtest: Backtest | None = None,
    base_config: dict | None = None,
    baseline: Strategy | None = None,
    reference: AblationArm | None = None,
    strategy_factory=None,
    progress=None,
    **backtest_kwargs,
) -> AblationResult:
    """Score every arm on one grid and compare each against the reference.

    A single :class:`Backtest` is shared across arms so prices load once and
    every arm is scored on exactly the same points — an arm scored on a
    different subset would not be comparable.

    Args:
        grid: the shared evaluation grid.
        arms: configurations to run. Duplicates by config are dropped.
        backtest: pre-built harness (mainly for tests and offline price data).
            When omitted one is built from ``backtest_kwargs``.
        base_config: config the arms' overrides are applied on top of.
        baseline: free comparison point for context; defaults to buy-and-hold.
        reference: arm every other arm is measured against. Defaults to the
            first arm, which the presets make the full pipeline.
        strategy_factory: builds a strategy from an arm. Defaults to
            :meth:`AblationArm.build`; override to ablate something the arm
            presets do not cover, or to run offline in tests. Whatever it
            returns must carry the arm's ``name``, or arms will collide in the
            decision cache.
        progress: optional ``(strategy_name, point, rating)`` callback.
    """
    arms = dedupe_arms(arms)
    if not arms:
        raise ValueError("run_ablation requires at least one arm")

    reference = reference or arms[0]
    if reference.name not in {a.name for a in arms}:
        arms = [reference, *arms]

    harness = backtest or Backtest(grid, **backtest_kwargs)
    harness.prepare()

    baseline = baseline or AlwaysRating("Buy")
    baseline_decisions = harness.run_strategy(baseline, progress=progress)

    build = strategy_factory or (lambda arm: arm.build(base_config))

    decisions: dict[str, list[Decision]] = {}
    for arm in arms:
        arm_decisions = harness.run_strategy(build(arm), progress=progress)
        if arm_decisions:
            decisions[arm.name] = arm_decisions
        else:
            logger.warning("Arm %r produced no scorable decisions; excluded", arm.label)

    if reference.name not in decisions:
        raise RuntimeError(
            f"reference arm {reference.label!r} produced no scorable decisions, so no "
            "arm can be compared against it"
        )

    metrics = {name: summarize(items) for name, items in decisions.items()}
    if baseline_decisions:
        metrics[baseline.name] = summarize(baseline_decisions)

    reference_decisions = decisions[reference.name]
    vs_reference = [
        compare(decisions[arm.name], reference_decisions, **_compare_kwargs(harness))
        for arm in arms
        if arm.name in decisions and arm.name != reference.name
    ]
    vs_baseline = (
        [
            compare(decisions[arm.name], baseline_decisions, **_compare_kwargs(harness))
            for arm in arms
            if arm.name in decisions
        ]
        if baseline_decisions
        else []
    )

    return AblationResult(
        grid=grid,
        arms=arms,
        reference=reference,
        metrics=metrics,
        vs_reference=vs_reference,
        vs_baseline=vs_baseline,
        decisions=decisions,
        skipped=list(harness.skipped),
    )


def _compare_kwargs(harness: Backtest) -> dict:
    """Mirror the harness's bootstrap settings so every comparison matches."""
    return {"iterations": harness.bootstrap_iterations, "seed": harness.seed}


def estimated_runs(grid: EvaluationGrid, arms: list[AblationArm]) -> int:
    """Full multi-agent runs a suite costs, before cache hits."""
    return len(dedupe_arms(arms)) * len(grid)
