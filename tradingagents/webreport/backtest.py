"""Render backtest and ablation results as charted HTML.

The markdown reports already say everything; this makes the important parts
visible at a glance. Two things drove the layout:

The **interval plot leads**, above the metrics. The claim this package exists to
support is "the difference from the baseline excludes zero", and that is a
statement about a picture. Putting the per-strategy return table first invites
the opposite reading — comparing absolute alphas across rows, which is exactly
the mistake the paired design is built to prevent.

**Every chart keeps its table.** That is not belt-and-braces: the neutral step of
the rating ramp sits below 3:1 against the surface in both themes, and the
palette's relief rule makes a table view or visible labels mandatory rather than
optional. Both are shipped.
"""

from __future__ import annotations

import re
from collections import defaultdict
from html import escape

from .charts import (
    RATING_ORDER,
    cumulative_chart,
    interval_chart,
    rating_chart,
    rating_legend,
    series_legend,
)
from .theme import render_page


def _fmt_pct(value: float, places: int = 3) -> str:
    return f"{value * 100:+.{places}f}%"


def _table(headers, rows, *, numeric_from: int = 1) -> str:
    head = "".join(
        f'<th class="{"num" if i >= numeric_from else ""}">{escape(str(h))}</th>'
        for i, h in enumerate(headers)
    )
    body = []
    for row in rows:
        cells = "".join(
            f'<td class="{"num" if i >= numeric_from else ""}">{cell}</td>'
            for i, cell in enumerate(row)
        )
        body.append(f"<tr>{cells}</tr>")
    return (
        f'<div class="scroll-x"><table><thead><tr>{head}</tr></thead>'
        f'<tbody>{"".join(body)}</tbody></table></div>'
    )


def _figure(svg: str, caption: str, legend_html: str = "") -> str:
    if not svg:
        return ""
    return (
        f'<figure class="card">{svg}{legend_html}'
        f"<figcaption>{caption}</figcaption></figure>"
    )


def _cumulative_series(decisions: dict) -> list[tuple[str, list[tuple[str, float]]]]:
    """Running mean alpha per strategy, in decision-date order.

    A running *mean* rather than a running sum: the sum grows with the number of
    decisions, so two strategies scored on different counts would not be
    comparable, and a late flat stretch would look like continued gains.
    """
    series = []
    for name, items in decisions.items():
        by_date = defaultdict(list)
        for decision in items:
            by_date[decision.point.date].append(decision.alpha_pnl)
        running_total = 0.0
        running_n = 0
        points = []
        for date in sorted(by_date):
            running_total += sum(by_date[date])
            running_n += len(by_date[date])
            points.append((date, running_total / running_n))
        series.append((name, points))
    return series


def _comparison_rows(comparisons, label_for=None):
    label_for = label_for or (lambda name: name)
    return [
        (label_for(c.baseline), c.mean_difference, c.ci_low, c.ci_high, c.significant)
        for c in comparisons
    ]


_BOLD_RE = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
_ITALIC_RE = re.compile(r"(?<!\*)\*([^*]+)\*(?!\*)")


def _verdict_block(text: str) -> str:
    """Render a verdict string, which is markdown, as HTML.

    The verdict wording is shared with the markdown reports so the two never
    drift. That means it arrives with ``**emphasis**`` in it, which would print
    literally if dropped into HTML — escape first, then convert the emphasis.
    """
    html = escape(text)
    html = _BOLD_RE.sub(r"<strong>\1</strong>", html)
    html = _ITALIC_RE.sub(r"<em>\1</em>", html)
    return f'<div class="verdict">{html}</div>'


def render_backtest_html(result, *, title: str = "Backtest Report") -> str:
    """Render a :class:`~tradingagents.backtest.harness.BacktestResult`."""
    from tradingagents.backtest.report import _verdict  # rendered text, shared wording

    grid = result.grid
    scored = len(next(iter(result.decisions.values())))
    subtitle = (
        f"{scored} scored decisions &middot; {len(grid.tickers)} tickers &middot; "
        f"{grid.start} to {grid.end} &middot; {grid.holding_days}-day hold &middot; "
        f"benchmark {escape(result.benchmark)}"
    )

    body = []

    if result.comparisons:
        body.append("<h2>Does it beat the baselines?</h2>")
        body.append(
            _figure(
                interval_chart(
                    _comparison_rows(result.comparisons),
                    title="Strategy minus baseline, mean alpha per decision",
                ),
                "Each bar is a 95% confidence interval on the paired difference. "
                "An interval crossing the zero line means no measurable edge over "
                "that baseline. Intervals are bootstrapped clustering on the "
                "decision date, since same-day decisions across tickers share a "
                "market factor.",
            )
        )
        body.append(
            _table(
                ["Baseline", "n", "Mean difference", "95% CI", "p", "Beats it?"],
                [
                    (
                        escape(c.baseline),
                        c.n,
                        _fmt_pct(c.mean_difference),
                        f"[{_fmt_pct(c.ci_low)}, {_fmt_pct(c.ci_high)}]",
                        f"{c.p_value:.3f}",
                        "<strong>yes</strong>"
                        if c.significant and c.mean_difference > 0
                        else "no",
                    )
                    for c in result.comparisons
                ],
            )
        )
        body.append(_verdict_block(_verdict(result.comparisons)))

    series = _cumulative_series(result.decisions)
    if series:
        body.append("<h2>Was any edge persistent?</h2>")
        body.append(
            _figure(
                cumulative_chart(
                    series,
                    title="Running mean alpha per decision",
                    y_label="running mean alpha",
                ),
                "Running mean alpha per decision as the grid progresses. A line "
                "that climbs once and then flattens made its result in one window; "
                "a steady grind is the shape a real edge has. Capped at four "
                "series, since the categorical palette assigns four hues and does "
                "not cycle.",
                series_legend(name for name, _ in series),
            )
        )

    body.append("<h2>Per-strategy metrics</h2>")
    body.append(
        _table(
            ["Strategy", "n", "Mean raw", "Mean alpha", "Hit rate", "IC", "t"],
            [
                (
                    escape(m.strategy),
                    m.n,
                    _fmt_pct(m.mean_raw),
                    _fmt_pct(m.mean_alpha),
                    f"{m.hit_rate:.1%}",
                    f"{m.information_coefficient:+.3f}",
                    f"{m.t_stat:+.2f}",
                )
                for m in result.metrics.values()
            ],
        )
    )
    body.append(
        '<p class="note">Alpha is the headline. Raw return mostly reflects '
        "whether the market rose over the window. The t column assumes "
        "independent observations, which the grid violates — trust the "
        "bootstrap intervals above.</p>"
    )

    body.append("<h2>Rating mix</h2>")
    body.append(
        _figure(
            rating_chart(
                [(m.strategy, m.rating_counts) for m in result.metrics.values()],
                title="Rating distribution by strategy",
            ),
            "A pipeline that emits Buy almost everywhere is buy-and-hold in "
            "costume, whatever its headline return says. Hold straddles the "
            "centre line so the axis reads as net direction.",
            rating_legend(),
        )
    )
    body.append(
        _table(
            ["Strategy", *RATING_ORDER],
            [
                (
                    escape(m.strategy),
                    *[
                        f"{m.rating_counts.get(r, 0)} "
                        f"({m.rating_counts.get(r, 0) / max(sum(m.rating_counts.values()), 1):.0%})"
                        for r in RATING_ORDER
                    ],
                )
                for m in result.metrics.values()
            ],
        )
    )

    if result.clean_metrics:
        body.append("<h2>Out-of-sample subset</h2>")
        body.append(
            '<p class="note">Points at or before the knowledge cutoff '
            f"({escape(str(grid.knowledge_cutoff))}) are excluded below. The "
            "model's weights encode what happened after any date before its "
            "training cutoff, so only these rows measure forecasting rather than "
            "recall.</p>"
        )
        if result.clean_comparisons:
            body.append(
                _figure(
                    interval_chart(
                        _comparison_rows(result.clean_comparisons),
                        title="Out-of-sample strategy minus baseline",
                    ),
                    "The same comparison restricted to uncontaminated dates. A "
                    "large gap against the chart above is evidence that apparent "
                    "skill came from recall.",
                )
            )
        body.append(
            _table(
                ["Strategy", "n", "Mean alpha", "Hit rate", "IC"],
                [
                    (
                        escape(m.strategy),
                        m.n,
                        _fmt_pct(m.mean_alpha),
                        f"{m.hit_rate:.1%}",
                        f"{m.information_coefficient:+.3f}",
                    )
                    for m in result.clean_metrics.values()
                ],
            )
        )
    elif not grid.knowledge_cutoff:
        body.append(
            '<p class="note"><strong>No knowledge cutoff was supplied</strong>, so '
            "nothing here is verified out-of-sample. If any evaluation date "
            "precedes the model's training cutoff, its result measures recall, "
            "not forecasting.</p>"
        )

    if result.skipped:
        body.append(
            f'<p class="note">{len(result.skipped)} grid point(s) were dropped for '
            "having no realized return or a failed run. They were dropped for "
            "every strategy alike, so the comparison stays paired.</p>"
        )

    return render_page(title, "\n".join(body), subtitle=subtitle)


def render_ablation_html(result, *, title: str = "Ablation Report") -> str:
    """Render an :class:`~tradingagents.backtest.ablation.AblationResult`."""
    from tradingagents.backtest.ablation_report import (
        _verdict,
        minimum_detectable_effect,
    )

    grid = result.grid
    scored = len(next(iter(result.decisions.values())))
    subtitle = (
        f"{len(result.arms)} arms &middot; {scored} scored decisions &middot; "
        f"{grid.start} to {grid.end} &middot; reference: "
        f"{escape(result.reference.label)}"
    )

    body = []

    if result.vs_reference:
        body.append("<h2>Arm minus reference</h2>")
        body.append(
            _figure(
                interval_chart(
                    [
                        (
                            result.label_for(c.strategy),
                            c.mean_difference,
                            c.ci_low,
                            c.ci_high,
                            c.significant,
                        )
                        for c in result.vs_reference
                    ],
                    title="Each arm versus the full pipeline",
                ),
                "An interval crossing zero means that arm did not measurably "
                "change the result. For an arm that <em>removes</em> work, that "
                "is a reason to drop the removed component, not to keep it.",
            )
        )
        mde = minimum_detectable_effect(result.vs_reference)
        body.append(
            f'<p class="note"><strong>Resolvable effect: about '
            f"{mde * 100:.3f}% per decision</strong> across "
            f"{len(grid.dates)} decision dates. A true difference smaller than "
            "that reads as &ldquo;no measurable difference&rdquo; here whether or "
            "not it is real — widen the date range or universe before concluding "
            "a component does nothing.</p>"
        )
        body.append(
            _table(
                ["Arm", "n", "Mean difference", "95% CI", "p", "Verdict"],
                [
                    (
                        escape(result.label_for(c.strategy)),
                        c.n,
                        _fmt_pct(c.mean_difference),
                        f"[{_fmt_pct(c.ci_low)}, {_fmt_pct(c.ci_high)}]",
                        f"{c.p_value:.3f}",
                        "no measurable difference"
                        if not c.significant
                        else ("<strong>better</strong>" if c.mean_difference > 0 else "<strong>worse</strong>"),
                    )
                    for c in result.vs_reference
                ],
            )
        )
        body.append(_verdict_block(_verdict(result)))

    body.append("<h2>Per-arm metrics</h2>")
    body.append(
        _table(
            ["Arm", "n", "Mean raw", "Mean alpha", "Hit rate", "IC"],
            [
                (
                    escape(result.label_for(name)),
                    m.n,
                    _fmt_pct(m.mean_raw),
                    _fmt_pct(m.mean_alpha),
                    f"{m.hit_rate:.1%}",
                    f"{m.information_coefficient:+.3f}",
                )
                for name, m in result.metrics.items()
            ],
        )
    )

    body.append("<h2>Rating mix by arm</h2>")
    body.append(
        _figure(
            rating_chart(
                [
                    (result.label_for(name), m.rating_counts)
                    for name, m in result.metrics.items()
                ],
                title="Rating distribution by arm",
            ),
            "Removing an analyst that shifts the rating mix but not the returns "
            "changed what the pipeline says without changing what it is worth.",
            rating_legend(),
        )
    )

    body.append(
        '<p class="note"><strong>Only analyst arms are structural.</strong> '
        "<code>selected_analysts</code> genuinely removes graph nodes. Debate and "
        "risk arms only change depth — the graph always runs at least one "
        "Bull/Bear exchange and one risk pass, so those arms cannot tell you what "
        "debate is worth versus none.</p>"
    )

    return render_page(title, "\n".join(body), subtitle=subtitle)
