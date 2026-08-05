"""Markdown rendering of a :class:`~tradingagents.backtest.harness.BacktestResult`.

The report is written to be hard to over-read. It leads with the paired
comparisons (which answer "did it beat the baseline?") rather than the strategy's
absolute return (which mostly answers "did the market go up?"), states the
significance verdict in words, and carries the methodology caveats inline so a
table pasted elsewhere brings its own health warnings.
"""

from __future__ import annotations

from .harness import BacktestResult
from .metrics import PairedComparison, StrategyMetrics


def render_report(result: BacktestResult) -> str:
    """Render a full markdown report for ``result``."""
    grid = result.grid
    lines = [
        "# Backtest Report",
        "",
        "## Setup",
        "",
        f"- **Evaluation points**: {len(grid)} "
        f"({len(grid.tickers)} tickers x {len(grid.dates)} dates)",
        f"- **Scored**: {len(next(iter(result.decisions.values())))} "
        f"(dropped {len(result.skipped)} with no realized return or a failed run)",
        f"- **Date range**: {grid.start} to {grid.end}",
        f"- **Holding period**: {grid.holding_days} trading days",
        f"- **Benchmark**: {result.benchmark}",
        f"- **Position map**: {result.position_map}",
        f"- **Universe**: {', '.join(grid.tickers)}",
    ]
    if grid.knowledge_cutoff:
        lines.append(
            f"- **Knowledge cutoff**: {grid.knowledge_cutoff} "
            f"({grid.contaminated_count} of {len(grid)} points at or before it)"
        )
    lines += ["", "## Headline: strategy vs baselines", ""]

    if result.comparisons:
        lines += _comparison_table(result.comparisons)
        lines += ["", _verdict(result.comparisons), ""]
    else:
        lines += ["No baseline comparisons were produced.", ""]

    lines += ["## Per-strategy metrics", "", *_metrics_table(result.metrics.values()), ""]
    lines += ["### Rating distribution", "", *_rating_table(result.metrics.values()), ""]

    if result.clean_metrics:
        lines += [
            "## Out-of-sample subset (after knowledge cutoff)",
            "",
            "Contaminated points are excluded below. The model's weights encode "
            "what happened after any date before its training cutoff, so only "
            "these rows measure forecasting rather than recall.",
            "",
            *_metrics_table(result.clean_metrics.values()),
            "",
        ]
        if result.clean_comparisons:
            lines += _comparison_table(result.clean_comparisons)
            lines += ["", _verdict(result.clean_comparisons), ""]

    lines += _caveats(result)
    return "\n".join(lines)


def _comparison_table(comparisons: list[PairedComparison]) -> list[str]:
    pct = int(comparisons[0].confidence * 100)
    rows = [
        f"| Baseline | n | Mean alpha difference | {pct}% CI | p | Beats baseline? |",
        "|---|---|---|---|---|---|",
    ]
    for c in comparisons:
        verdict = "**yes**" if c.significant and c.mean_difference > 0 else "no"
        rows.append(
            f"| {c.baseline} | {c.n} | {c.mean_difference * 100:+.3f}% | "
            f"[{c.ci_low * 100:+.3f}%, {c.ci_high * 100:+.3f}%] | {c.p_value:.3f} | {verdict} |"
        )
    return rows


def _metrics_table(metrics) -> list[str]:
    rows = [
        "| Strategy | n | Dates | Mean raw | Mean alpha | Hit rate | IC | t |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for m in metrics:
        rows.append(
            f"| {m.strategy} | {m.n} | {m.n_dates} | {m.mean_raw * 100:+.3f}% | "
            f"{m.mean_alpha * 100:+.3f}% | {m.hit_rate:.1%} | "
            f"{m.information_coefficient:+.3f} | {m.t_stat:+.2f} |"
        )
    return rows


def _rating_table(metrics) -> list[str]:
    """Rating mix per strategy — the fastest way to spot a degenerate pipeline.

    A strategy that emits Buy almost everywhere is buy-and-hold wearing a
    costume, whatever its headline return says.
    """
    from tradingagents.agents.utils.rating import RATINGS_5_TIER

    rows = ["| Strategy | " + " | ".join(RATINGS_5_TIER) + " |", "|---" * 6 + "|"]
    for m in metrics:
        total = max(sum(m.rating_counts.values()), 1)
        cells = [
            f"{m.rating_counts.get(r, 0)} ({m.rating_counts.get(r, 0) / total:.0%})"
            for r in RATINGS_5_TIER
        ]
        rows.append(f"| {m.strategy} | " + " | ".join(cells) + " |")
    return rows


def _verdict(comparisons: list[PairedComparison]) -> str:
    """One sentence stating what the comparisons collectively support."""
    wins = [c for c in comparisons if c.significant and c.mean_difference > 0]
    losses = [c for c in comparisons if c.significant and c.mean_difference < 0]
    if len(wins) == len(comparisons):
        return (
            "**Verdict**: the strategy beats every baseline by a margin whose "
            "confidence interval excludes zero."
        )
    if wins:
        beaten = ", ".join(c.baseline for c in wins)
        return (
            f"**Verdict**: the strategy significantly beats {beaten}, but not every "
            "baseline. Partial evidence of edge."
        )
    if losses:
        lost = ", ".join(c.baseline for c in losses)
        return (
            f"**Verdict**: the strategy significantly *underperforms* {lost}. "
            "No evidence of edge."
        )
    return (
        "**Verdict**: no comparison's confidence interval excludes zero. The "
        "results are consistent with the strategy having no edge over the "
        "baselines — which, given the cost of running it, is the finding."
    )


def _caveats(result: BacktestResult) -> list[str]:
    lines = [
        "## Reading this honestly",
        "",
        "- **Alpha, not raw return, is the number that matters.** Raw return "
        "mostly reflects whether the market rose over the window.",
        "- **Confidence intervals are bootstrapped clustering on the decision "
        "date**, because same-day decisions across tickers share a market "
        "factor and are not independent observations.",
        "- **The universes are survivorship-biased** (they are names that exist "
        "and are liquid today), which lifts absolute returns. Paired "
        "differences against baselines on the identical grid are unaffected.",
    ]
    if "random_matched" in result.metrics:
        lines.append(
            "- **`random_matched` is the strictest baseline.** It replays the "
            "strategy's own rating distribution in random order, so beating it "
            "requires the ratings to be assigned correctly, not merely to be "
            "bullish on average. Losing to it while beating `always_buy` means "
            "the only thing the pipeline contributed was a net-long tilt."
        )
    if result.grid.knowledge_cutoff:
        lines.append(
            "- **Points at or before the knowledge cutoff are not out-of-sample.** "
            "The model may recall the outcome. Compare the headline table "
            "against the out-of-sample subset above."
        )
    else:
        lines.append(
            "- **No knowledge cutoff was supplied**, so nothing here is verified "
            "out-of-sample. If any evaluation date precedes the model's training "
            "cutoff, its result measures recall, not forecasting. Re-run with "
            "`--knowledge-cutoff`."
        )
    if result.skipped:
        lines += [
            "",
            f"### Dropped points ({len(result.skipped)})",
            "",
            *[f"- `{key}`: {reason}" for key, reason in result.skipped[:20]],
        ]
        if len(result.skipped) > 20:
            lines.append(f"- ... and {len(result.skipped) - 20} more")
    return lines


def render_summary(result: BacktestResult) -> str:
    """Compact console summary: the comparison table and the verdict."""
    parts = [
        f"{len(result.grid)} points | {result.grid.start}..{result.grid.end} | "
        f"{result.grid.holding_days}d hold | benchmark {result.benchmark}",
        "",
        *_metrics_table(result.metrics.values()),
    ]
    if result.comparisons:
        parts += ["", *_comparison_table(result.comparisons), "", _verdict(result.comparisons)]
    return "\n".join(parts)


def summarize_strategy(metrics: StrategyMetrics) -> str:
    """One-line description of a single strategy's performance."""
    return (
        f"{metrics.strategy}: n={metrics.n} "
        f"alpha={metrics.mean_alpha * 100:+.3f}% "
        f"hit={metrics.hit_rate:.1%} IC={metrics.information_coefficient:+.3f}"
    )
