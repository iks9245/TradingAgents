"""Markdown rendering for :class:`~tradingagents.backtest.ablation.AblationResult`.

The table that matters is "arm minus reference". Its verdict column is worded to
resist the most likely misreading: a difference whose interval spans zero is not
evidence that the removed component helps, it is the absence of evidence that it
does anything at all — and for a component that costs tokens, that asymmetry is
the point.
"""

from __future__ import annotations

from .ablation import AblationResult
from .metrics import PairedComparison


def render_ablation_report(result: AblationResult) -> str:
    """Full markdown report for an ablation run."""
    grid = result.grid
    scored = len(next(iter(result.decisions.values())))
    lines = [
        "# Ablation Report",
        "",
        "## Setup",
        "",
        f"- **Reference arm**: {result.reference.label} (`{result.reference.name}`)",
        f"- **Arms**: {len(result.arms)}",
        f"- **Evaluation points**: {len(grid)} "
        f"({len(grid.tickers)} tickers x {len(grid.dates)} dates), {scored} scored",
        f"- **Date range**: {grid.start} to {grid.end}",
        f"- **Holding period**: {grid.holding_days} trading days",
        f"- **Universe**: {', '.join(grid.tickers)}",
    ]
    if grid.knowledge_cutoff:
        lines.append(
            f"- **Knowledge cutoff**: {grid.knowledge_cutoff} "
            f"({grid.contaminated_count} of {len(grid)} points at or before it)"
        )

    lines += ["", "## Arm minus reference", ""]
    if result.vs_reference:
        lines += _difference_table(result, result.vs_reference)
        lines += ["", _power_note(result), "", _verdict(result), ""]
    else:
        lines += ["Only the reference arm ran; nothing to compare.", ""]

    lines += ["## Per-arm metrics", "", *_metrics_table(result), ""]

    if result.vs_baseline:
        lines += [
            "## Each arm vs buy-and-hold",
            "",
            "Context, not the headline. An arm can differ from the reference "
            "while neither beats doing nothing.",
            "",
            *_difference_table(result, result.vs_baseline, against="baseline"),
            "",
        ]

    lines += _caveats(result)
    return "\n".join(lines)


def _difference_table(
    result: AblationResult, comparisons: list[PairedComparison], against: str = "reference"
) -> list[str]:
    pct = int(comparisons[0].confidence * 100)
    header = "Arm" if against == "reference" else "Arm"
    rows = [
        f"| {header} | n | Mean alpha difference | {pct}% CI | p | Verdict |",
        "|---|---|---|---|---|---|",
    ]
    for c in comparisons:
        label = result.label_for(c.strategy)
        if not c.significant:
            verdict = "no measurable difference"
        elif c.mean_difference > 0:
            verdict = "**better**"
        else:
            verdict = "**worse**"
        rows.append(
            f"| {label} | {c.n} | {c.mean_difference * 100:+.3f}% | "
            f"[{c.ci_low * 100:+.3f}%, {c.ci_high * 100:+.3f}%] | {c.p_value:.3f} | {verdict} |"
        )
    return rows


def _metrics_table(result: AblationResult) -> list[str]:
    rows = [
        "| Arm | n | Mean raw | Mean alpha | Hit rate | IC | Buy % |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, m in result.metrics.items():
        total = max(sum(m.rating_counts.values()), 1)
        buy_share = m.rating_counts.get("Buy", 0) / total
        rows.append(
            f"| {result.label_for(name)} | {m.n} | {m.mean_raw * 100:+.3f}% | "
            f"{m.mean_alpha * 100:+.3f}% | {m.hit_rate:.1%} | "
            f"{m.information_coefficient:+.3f} | {buy_share:.0%} |"
        )
    return rows


def minimum_detectable_effect(comparisons: list[PairedComparison]) -> float:
    """Smallest per-decision alpha difference this run could have resolved.

    The confidence interval's half-width: an effect smaller than this lands
    inside the interval and reads as "no measurable difference" no matter how
    real it is. Reported so an underpowered null is not mistaken for evidence
    that a component does nothing — the two look identical in the table.
    """
    if not comparisons:
        return 0.0
    half_widths = sorted((c.ci_high - c.ci_low) / 2.0 for c in comparisons)
    mid = len(half_widths) // 2
    if len(half_widths) % 2:
        return half_widths[mid]
    return (half_widths[mid - 1] + half_widths[mid]) / 2.0


def _power_note(result: AblationResult) -> str:
    mde = minimum_detectable_effect(result.vs_reference)
    n_dates = result.grid and len(result.grid.dates)
    return (
        f"**Resolvable effect**: roughly {mde * 100:.3f}% per decision "
        f"({n_dates} decision dates). A true difference smaller than that will "
        "read as \"no measurable difference\" here regardless of whether it is "
        "real — widen the date range or the universe before concluding a "
        "component does nothing."
    )


def _verdict(result: AblationResult) -> str:
    """One sentence on what the arm-vs-reference differences collectively show."""
    moved = [c for c in result.vs_reference if c.significant]
    if not moved:
        return (
            "**Verdict**: no arm differs measurably from the full pipeline. On "
            "this grid, the ablated components did not change the result — for "
            "components that cost tokens, that is a reason to drop them, not a "
            "reason to keep them."
        )
    better = [c for c in moved if c.mean_difference > 0]
    worse = [c for c in moved if c.mean_difference < 0]
    parts = []
    if worse:
        parts.append(
            "removing " + ", ".join(result.label_for(c.strategy) for c in worse)
            + " measurably hurts, so those components are earning their cost"
        )
    if better:
        parts.append(
            ", ".join(result.label_for(c.strategy) for c in better)
            + " measurably *beats* the full pipeline, which means the removed work was "
            "actively harmful on this grid"
        )
    return "**Verdict**: " + "; ".join(parts) + "."


def _caveats(result: AblationResult) -> list[str]:
    lines = [
        "## Reading this honestly",
        "",
        "- **A null result is informative here.** An arm that removes work and "
        "shows no measurable difference means that work was not paying for "
        "itself on this grid — not that it is valuable but hard to detect.",
        "- **Absence of a difference is not proof of equivalence.** With few "
        "evaluation points the interval is wide enough to hide a real effect. "
        "Check the interval width against the effect size you would care about "
        "before concluding anything.",
        "- **Only analyst arms are structural.** `selected_analysts` genuinely "
        "removes graph nodes. Debate and risk arms only change *depth*: the "
        "graph always runs at least one Bull/Bear exchange and one pass through "
        "the risk analysts, so those arms cannot tell you what debate is worth "
        "versus none.",
        "- **Every arm was scored on the identical points**, and comparisons are "
        "paired with the bootstrap clustering on the decision date.",
    ]
    if result.grid.knowledge_cutoff:
        lines.append(
            "- **Contaminated points are included above.** Ablation differences "
            "are less sensitive to knowledge contamination than absolute "
            "performance is — every arm shares the same leak — but an arm that "
            "changes what the model recalls is not fully controlled for."
        )
    else:
        lines.append(
            "- **No knowledge cutoff was supplied.** Ablation differences are "
            "more robust to this than absolute returns, since the leak is common "
            "to every arm, but pass `--knowledge-cutoff` if you also want the "
            "absolute numbers to mean something."
        )
    if result.skipped:
        lines += ["", f"### Dropped points ({len(result.skipped)})", ""]
        lines += [f"- `{key}`: {reason}" for key, reason in result.skipped[:20]]
        if len(result.skipped) > 20:
            lines.append(f"- ... and {len(result.skipped) - 20} more")
    return lines


def render_ablation_summary(result: AblationResult) -> str:
    """Compact console summary."""
    parts = [
        f"{len(result.arms)} arms | {len(result.grid)} points | "
        f"{result.grid.start}..{result.grid.end} | reference: {result.reference.label}",
        "",
        *_metrics_table(result),
    ]
    if result.vs_reference:
        parts += [
            "",
            *_difference_table(result, result.vs_reference),
            "",
            _power_note(result),
            "",
            _verdict(result),
        ]
    return "\n".join(parts)
