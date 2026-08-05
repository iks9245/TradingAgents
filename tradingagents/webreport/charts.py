"""Inline SVG charts, drawn from strings with no plotting dependency.

Three forms, each picked for the job its data actually does:

**Confidence intervals** (:func:`interval_chart`) — the headline. The statistical
claim in this whole package is "does the interval exclude zero", and a plot makes
that a matter of looking rather than of reading two numbers and comparing them
mentally. Position against the zero line carries the meaning; colour reinforces
it and never carries it alone.

**Cumulative alpha** (:func:`cumulative_chart`) — change over time, so a line.
It answers the question a single mean cannot: was the edge steady, or one lucky
window? A strategy whose whole result comes from two weeks is not the same
finding as one that grinds upward.

**Rating mix** (:func:`rating_chart`) — the five-tier scale is ordered and
signed, so it takes a diverging bar around a neutral centre, not categorical
hues. Bullish extends right, bearish left. A pipeline that emits Buy everywhere
is instantly a solid block on one side.

All three are static SVG with native ``<title>`` hover, so they work in email
clients, print, and any browser without scripting.
"""

from __future__ import annotations

import math
from html import escape

# Geometry, in SVG user units. The viewBox scales to the container, so these are
# proportions rather than pixels.
_W = 720
_ROW_H = 34
_PAD_L = 168
_PAD_R = 78
_PAD_T = 34
_PAD_B = 40

RATING_ORDER = ("Buy", "Overweight", "Hold", "Underweight", "Sell")


def _fmt_pct(value: float, places: int = 2) -> str:
    return f"{value * 100:+.{places}f}%"


def _nice_bounds(low: float, high: float) -> tuple[float, float]:
    """Symmetric-ish axis bounds with padding, always including zero."""
    low = min(low, 0.0)
    high = max(high, 0.0)
    span = high - low
    if span <= 0:
        return -0.01, 0.01
    pad = span * 0.12
    return low - pad, high + pad


def _scale(value: float, lo: float, hi: float, px0: float, px1: float) -> float:
    if hi == lo:
        return (px0 + px1) / 2
    return px0 + (value - lo) / (hi - lo) * (px1 - px0)


def _axis_ticks(lo: float, hi: float, count: int = 5) -> list[float]:
    """Ticks on a "nice" step, aligned so zero falls on one of them.

    Aligning to zero rather than appending it is what keeps the labels apart: an
    appended zero sits an arbitrary distance from its neighbours and their text
    collides ("-0.4%" printed over "0.0%"). Choosing a 1/2/2.5/5 x 10^k step and
    laying ticks at multiples of it puts zero on the grid by construction.
    """
    span = hi - lo
    if span <= 0:
        return [lo]
    raw = span / max(count - 1, 1)
    magnitude = 10.0 ** math.floor(math.log10(raw))
    for multiple in (1, 2, 2.5, 5, 10):
        step = magnitude * multiple
        if step >= raw:
            break
    first = math.ceil(lo / step) * step
    ticks = []
    value = first
    while value <= hi + step * 1e-9:
        # Snap values that are zero up to float error, so the label reads "0.0%"
        # and the zero-line test matches exactly.
        ticks.append(0.0 if abs(value) < step * 1e-9 else value)
        value += step
    return ticks


def _text_width(text: str, font_px: float) -> float:
    """Rough advance width for a label, used to size margins.

    SVG cannot measure text before layout, and anything wider than the viewBox
    is silently clipped rather than overflowing visibly — so a long strategy
    name loses its tail and reads as a different, shorter name. Estimating high
    (0.58em average for this UI sans) costs a little whitespace and avoids that.
    """
    return len(text) * font_px * 0.58


def _spread_labels(positions: list[float], min_gap: float) -> list[float]:
    """Nudge label y-positions apart, preserving their original order.

    Line ends converge when strategies finish near the same value — which is
    exactly what a null result looks like — so the labels that identify them
    collide precisely when the chart matters most.
    """
    order = sorted(range(len(positions)), key=lambda i: positions[i])
    adjusted = list(positions)
    for rank, idx in enumerate(order):
        if rank == 0:
            continue
        previous = adjusted[order[rank - 1]]
        if adjusted[idx] - previous < min_gap:
            adjusted[idx] = previous + min_gap
    return adjusted


def interval_chart(
    rows,
    *,
    title: str = "",
    value_label: str = "mean alpha difference",
) -> str:
    """Horizontal confidence intervals against a zero line.

    Args:
        rows: sequence of ``(label, point_estimate, ci_low, ci_high, significant)``.
            Significance drives colour and the accompanying marker, but the row
            label and the table beneath the chart both state it in words, so the
            reading never depends on colour.
        title: accessible name for the figure.
        value_label: axis caption.
    """
    rows = list(rows)
    if not rows:
        return ""

    height = _PAD_T + _ROW_H * len(rows) + _PAD_B
    lo, hi = _nice_bounds(
        min(r[2] for r in rows), max(r[3] for r in rows)
    )
    # Grow the left gutter for long labels rather than letting them clip.
    pad_l = max(_PAD_L, min(320.0, max(_text_width(str(r[0]), 12.5) for r in rows) + 22))
    x0, x1 = pad_l, _W - _PAD_R

    parts = [
        f'<svg class="chart" viewBox="0 0 {_W} {height}" role="img" '
        f'aria-label="{escape(title or value_label)}" xmlns="http://www.w3.org/2000/svg">'
    ]
    if title:
        parts.append(f"<title>{escape(title)}</title>")

    # Gridlines and tick labels first, so marks sit above them.
    for tick in _axis_ticks(lo, hi):
        x = _scale(tick, lo, hi, x0, x1)
        is_zero = abs(tick) < 1e-12
        cls = "zero-line" if is_zero else "grid-line"
        parts.append(
            f'<line class="{cls}" x1="{x:.1f}" y1="{_PAD_T - 10:.1f}" '
            f'x2="{x:.1f}" y2="{height - _PAD_B + 6:.1f}"/>'
        )
        parts.append(
            f'<text class="tick" x="{x:.1f}" y="{height - _PAD_B + 22:.1f}" '
            f'text-anchor="middle">{_fmt_pct(tick, 1)}</text>'
        )

    for i, (label, point, ci_low, ci_high, significant) in enumerate(rows):
        cy = _PAD_T + _ROW_H * i + _ROW_H / 2
        bar_x0 = _scale(ci_low, lo, hi, x0, x1)
        bar_x1 = _scale(ci_high, lo, hi, x0, x1)
        px = _scale(point, lo, hi, x0, x1)

        if not significant:
            colour = "var(--text-muted)"
            verdict = "interval includes zero"
        elif point > 0:
            colour = "var(--positive)"
            verdict = "excludes zero, positive"
        else:
            colour = "var(--negative)"
            verdict = "excludes zero, negative"

        tip = (
            f"{label}: {_fmt_pct(point)} "
            f"[{_fmt_pct(ci_low)}, {_fmt_pct(ci_high)}] — {verdict}"
        )
        parts.append(f'<g class="mark"><title>{escape(tip)}</title>')
        # 4px rounded ends on the interval bar; thin so the dot stays dominant.
        parts.append(
            f'<rect x="{min(bar_x0, bar_x1):.1f}" y="{cy - 3:.1f}" '
            f'width="{abs(bar_x1 - bar_x0):.1f}" height="6" rx="3" '
            f'fill="{colour}" opacity="0.34"/>'
        )
        # Whiskers make the interval ends legible when the bar is very short.
        for edge in (bar_x0, bar_x1):
            parts.append(
                f'<line x1="{edge:.1f}" y1="{cy - 7:.1f}" x2="{edge:.1f}" '
                f'y2="{cy + 7:.1f}" stroke="{colour}" stroke-width="1.5" opacity="0.7"/>'
            )
        # Point estimate: >= 8px, with a surface ring so it stays readable where
        # it overlaps the bar or a gridline.
        parts.append(
            f'<circle cx="{px:.1f}" cy="{cy:.1f}" r="5" fill="{colour}" '
            f'stroke="var(--surface)" stroke-width="2"/>'
        )
        parts.append("</g>")

        parts.append(
            f'<text class="row-label" x="{pad_l - 12:.1f}" y="{cy + 4:.1f}" '
            f'text-anchor="end">{escape(str(label))}</text>'
        )
        parts.append(
            f'<text class="value-label" x="{_W - _PAD_R + 10:.1f}" y="{cy + 4:.1f}">'
            f"{_fmt_pct(point)}</text>"
        )

    parts.append(
        f'<text class="tick" x="{(x0 + x1) / 2:.1f}" y="{height - 4:.1f}" '
        f'text-anchor="middle">{escape(value_label)}</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts)


def cumulative_chart(series, *, title: str = "", y_label: str = "cumulative alpha") -> str:
    """Multi-series line chart of a running total over ordered dates.

    Args:
        series: sequence of ``(name, [(date_str, cumulative_value), ...])``.
            Capped at four series — the palette assigns four categorical hues in
            fixed order and does not cycle. A fifth would have to fold into an
            "other" bucket or become a small multiple.
    """
    series = [s for s in series if s[1]]
    if not series:
        return ""
    series = series[:4]

    height = 300
    # Direct labels sit to the right of the last point; reserve real space for
    # them or the longest name is clipped and reads as a shorter one.
    label_gutter = max(_text_width(name, 12) for name, _ in series) + 24
    x0, x1 = 58, _W - max(108.0, min(300.0, label_gutter))
    y0, y1 = _PAD_T, height - 46

    n_points = max(len(points) for _, points in series)
    values = [v for _, points in series for _, v in points]
    lo, hi = _nice_bounds(min(values), max(values))

    parts = [
        f'<svg class="chart" viewBox="0 0 {_W} {height}" role="img" '
        f'aria-label="{escape(title or y_label)}" xmlns="http://www.w3.org/2000/svg">'
    ]
    if title:
        parts.append(f"<title>{escape(title)}</title>")

    for tick in _axis_ticks(lo, hi):
        y = _scale(tick, lo, hi, y1, y0)
        cls = "zero-line" if abs(tick) < 1e-12 else "grid-line"
        parts.append(f'<line class="{cls}" x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}"/>')
        parts.append(
            f'<text class="tick" x="{x0 - 8}" y="{y + 4:.1f}" text-anchor="end">'
            f"{_fmt_pct(tick, 1)}</text>"
        )

    end_y = [_scale(points[-1][1], lo, hi, y1, y0) for _, points in series]
    label_y = _spread_labels(end_y, min_gap=15)

    for idx, (name, points) in enumerate(series):
        colour = f"var(--series-{idx + 1})"
        coords = []
        for i, (_, value) in enumerate(points):
            x = _scale(i, 0, max(n_points - 1, 1), x0, x1)
            y = _scale(value, lo, hi, y1, y0)
            coords.append(f"{x:.1f},{y:.1f}")
        parts.append(
            f'<polyline class="series-line" points="{" ".join(coords)}" stroke="{colour}"/>'
        )
        # Direct label at the line end: with <= 4 series every one is labelled,
        # so identity never rests on the legend swatch alone.
        last_x = _scale(len(points) - 1, 0, max(n_points - 1, 1), x0, x1)
        last_y = end_y[idx]
        parts.append(
            f'<circle cx="{last_x:.1f}" cy="{last_y:.1f}" r="4" fill="{colour}" '
            f'stroke="var(--surface)" stroke-width="2"><title>'
            f"{escape(name)}: {_fmt_pct(points[-1][1])}</title></circle>"
        )
        text_y = label_y[idx]
        # A leader line where the label had to move, so it still reads as
        # belonging to its own line rather than a neighbouring one.
        if abs(text_y - last_y) > 1:
            parts.append(
                f'<line x1="{last_x + 4:.1f}" y1="{last_y:.1f}" x2="{last_x + 8:.1f}" '
                f'y2="{text_y:.1f}" stroke="{colour}" stroke-width="1" opacity="0.5"/>'
            )
        parts.append(
            f'<text class="value-label" x="{last_x + 10:.1f}" y="{text_y + 4:.1f}" '
            f'fill="{colour}">{escape(name)}</text>'
        )

    first_date = series[0][1][0][0]
    last_date = series[0][1][-1][0]
    parts.append(f'<text class="tick" x="{x0}" y="{height - 16}">{escape(first_date)}</text>')
    parts.append(
        f'<text class="tick" x="{x1}" y="{height - 16}" text-anchor="end">'
        f"{escape(last_date)}</text>"
    )
    parts.append(
        f'<text class="tick" x="{(x0 + x1) / 2:.1f}" y="{height - 2}" '
        f'text-anchor="middle">{escape(y_label)} by decision date</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts)


def rating_chart(rows, *, title: str = "") -> str:
    """Diverging bars of rating mix: bullish right, bearish left, Hold centred.

    Args:
        rows: sequence of ``(label, {rating: count})``.

    Hold straddles the centre line, half to each side, which keeps the axis
    meaning "net direction" rather than silently assigning neutral votes to one
    camp. Segments at least a few percent wide carry a visible percentage label
    — required relief, since the neutral step sits below 3:1 against the surface.
    """
    rows = [(label, counts) for label, counts in rows if sum(counts.values())]
    if not rows:
        return ""

    height = _PAD_T + _ROW_H * len(rows) + _PAD_B
    pad_l = max(_PAD_L, min(320.0, max(_text_width(str(r[0]), 12.5) for r in rows) + 22))
    x0, x1 = pad_l, _W - 40
    mid = (x0 + x1) / 2
    half = (x1 - x0) / 2

    parts = [
        f'<svg class="chart" viewBox="0 0 {_W} {height}" role="img" '
        f'aria-label="{escape(title or "rating distribution")}" '
        f'xmlns="http://www.w3.org/2000/svg">'
    ]
    if title:
        parts.append(f"<title>{escape(title)}</title>")

    for i, (label, counts) in enumerate(rows):
        total = sum(counts.values())
        cy = _PAD_T + _ROW_H * i + _ROW_H / 2
        bar_h = 18

        bullish = [("Buy", counts.get("Buy", 0)), ("Overweight", counts.get("Overweight", 0))]
        bearish = [("Sell", counts.get("Sell", 0)), ("Underweight", counts.get("Underweight", 0))]
        hold = counts.get("Hold", 0)

        # Rightward: half of Hold, then Overweight, then Buy at the outside.
        cursor = mid + (hold / 2 / total) * half
        segments = []
        for name, count in reversed(bullish):
            if not count:
                continue
            width = count / total * half
            segments.append((name, count, cursor, width))
            cursor += width
        # Leftward, mirrored.
        cursor = mid - (hold / 2 / total) * half
        for name, count in reversed(bearish):
            if not count:
                continue
            width = count / total * half
            segments.append((name, count, cursor - width, width))
            cursor -= width
        if hold:
            width = hold / total * half
            segments.append(("Hold", hold, mid - width / 2, width))

        for name, count, seg_x, width in segments:
            share = count / total
            colour = f"var(--rating-{name.lower()})"
            # 2px surface gap between adjacent fills keeps segment boundaries
            # readable without a stroke that would darken thin segments.
            draw_w = max(width - 2, 1)
            parts.append(
                f'<g class="mark"><title>{escape(label)} — {name}: '
                f"{count} ({share:.0%})</title>"
                f'<rect x="{seg_x + 1:.1f}" y="{cy - bar_h / 2:.1f}" '
                f'width="{draw_w:.1f}" height="{bar_h}" rx="3" fill="{colour}"/></g>'
            )
            if share >= 0.09:
                parts.append(
                    f'<text class="value-label" x="{seg_x + width / 2:.1f}" '
                    f'y="{cy + 4:.1f}" text-anchor="middle" '
                    f'style="font-size:11px">{share:.0%}</text>'
                )

        parts.append(
            f'<text class="row-label" x="{x0 - 12:.1f}" y="{cy + 4:.1f}" '
            f'text-anchor="end">{escape(str(label))}</text>'
        )

    parts.append(
        f'<line class="zero-line" x1="{mid:.1f}" y1="{_PAD_T - 8:.1f}" '
        f'x2="{mid:.1f}" y2="{height - _PAD_B + 4:.1f}"/>'
    )
    parts.append(
        f'<text class="tick" x="{mid:.1f}" y="{height - _PAD_B + 20:.1f}" '
        f'text-anchor="middle">bearish &#8592; net direction &#8594; bullish</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts)


def legend(items) -> str:
    """Swatch legend. Always rendered for two or more series."""
    entries = "".join(
        f'<li><span class="swatch" style="background:{colour}"></span>{escape(str(name))}</li>'
        for name, colour in items
    )
    return f'<ul class="legend">{entries}</ul>'


def rating_legend() -> str:
    return legend(
        (name, f"var(--rating-{name.lower()})") for name in RATING_ORDER
    )


def series_legend(names) -> str:
    return legend(
        (name, f"var(--series-{i + 1})") for i, name in enumerate(list(names)[:4])
    )
