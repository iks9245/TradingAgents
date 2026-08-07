"""Deterministic numeric consistency checks for assembled markdown reports."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Finding:
    """A numeric inconsistency found in the completed report."""

    kind: str
    summary: str
    detail: str


# Keep this deliberately narrow: report snapshots use ordinary decimal numbers,
# and accepting partial or malformed values would make a warning untrustworthy.
_NUMBER = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"

# The trailing guard rejects a digit (so a number is never cut short) and a
# comma that starts a thousands group (so "1,234" is never read as "1"). It must
# NOT reject an ordinary sentence comma: with a blanket `(?![\d,])` the text
# "at $111.52, confirming" failed on the full match, backtracked, and matched
# "111" — which then "conflicted" with the 111.52 stated elsewhere. Every bogus
# warning on the 2026-08-06 INTC report came from that self-inflicted mismatch.
_NUMBER_RE = re.compile(rf"(?<![\d,])({_NUMBER})(?!\d|,\d{{3}})")


def _parse_number(value: str) -> float | None:
    """Parse the report's supported decimal notation, or return ``None``."""
    if not re.fullmatch(_NUMBER, value):
        return None
    try:
        return float(value.replace(",", ""))
    except (TypeError, ValueError, OverflowError):
        return None


# Financial prose writes a negative amount as "-$2.54B", putting the currency
# symbol between the sign and the digits. The number pattern only accepts a sign
# glued to the digits, so such a value parsed as +2.54 — and a free cash flow of
# -2.54 stopped matching the same figure quoted elsewhere. Requiring the
# currency mark keeps this away from ordinary hyphens like "1-1.5x ATR".
_DETACHED_MINUS_RE = re.compile(r"[-−–]\s*[$¥€£]\s*$")


def _apply_detached_sign(value: float, prefix: str) -> float:
    """Negate ``value`` when the text before it carries a detached minus sign."""
    return -abs(value) if _DETACHED_MINUS_RE.search(prefix) else value


def _display_number(value: float) -> str:
    """Use a compact, stable display form in reader-facing evidence."""
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _within_rounding_tolerance(stated: float, computed: float) -> bool:
    return abs(stated - computed) <= max(0.05, 0.005 * abs(computed))


def _arithmetic_finding(
    stated: float,
    computed: float,
    left: float,
    right: float,
    suffix: str,
    expression: str,
) -> Finding | None:
    if _within_rounding_tolerance(stated, computed):
        return None
    stated_text = f"{stated:.2f}{suffix}"
    computed_text = f"{computed:.2f}{suffix}"
    difference = abs(stated - computed)
    unit = "percentage points" if suffix == "%" else "multiple units"
    return Finding(
        kind="arithmetic",
        summary=f"Stated {stated_text} but {expression} = {computed_text}",
        detail=(
            f"Operands: {_display_number(left)} and {_display_number(right)}. "
            f"Stated value: {stated_text}; computed value: {computed_text}. "
            f"Difference: {difference:.2f} {unit}."
        ),
    )


def _division_findings(markdown: str) -> list[Finding]:
    """Recompute copied ratio forms, catching mangled operating-margin prose."""
    findings: list[Finding] = []
    number = _NUMBER
    patterns = (
        # The order matters: the percent-change form must not be treated as a
        # plain division, because its "- 1" is the difference between growth
        # and a level ratio.
        (
            re.compile(
                rf"(?<![\d,])(?P<stated>{number})%(?![\d,])\s*"
                rf"\(\s*=\s*(?P<left>{number})\s*/\s*(?P<right>{number})\s*-\s*1\s*\)"
            ),
            "%",
            True,
        ),
        (
            re.compile(
                rf"(?<![\d,])(?P<stated>{number})%(?![\d,])\s*"
                rf"\(\s*=\s*(?P<multiple>{number})\s*[xX]\s*\)"
            ),
            "%",
            None,
        ),
        (
            re.compile(
                rf"(?<![\d,])(?P<stated>{number})%(?![\d,])\s*"
                rf"\(\s*(?P<left>{number})\s*/\s*(?P<right>{number})\s*\)"
            ),
            "%",
            False,
        ),
        (
            re.compile(
                rf"(?<![\d,])(?P<stated>{number})\s*[xX](?![\w])\s*"
                rf"\(\s*(?P<left>{number})\s*/\s*(?P<right>{number})\s*\)"
            ),
            "x",
            False,
        ),
        (
            re.compile(
                rf"(?<![\d,])(?P<left>{number})\s*/\s*(?P<right>{number})\s*=\s*"
                rf"(?P<stated>{number})(?P<suffix>%|[xX])(?![\w])"
            ),
            "generic",
            False,
        ),
    )

    for pattern, suffix, is_percent_change in patterns:
        for match in pattern.finditer(markdown):
            stated = _parse_number(match.group("stated"))
            if stated is None:
                continue

            if suffix == "%" and is_percent_change is None:
                multiple = _parse_number(match.group("multiple"))
                if multiple is None:
                    continue
                finding = _arithmetic_finding(
                    stated,
                    multiple * 100,
                    multiple,
                    1,
                    "%",
                    f"{_display_number(multiple)}x × 100",
                )
            else:
                left = _parse_number(match.group("left"))
                right = _parse_number(match.group("right"))
                if left is None or right is None or right == 0:
                    continue
                actual_suffix = match.group("suffix") if suffix == "generic" else suffix
                computed = left / right
                if actual_suffix == "%":
                    computed = (computed - 1) * 100 if is_percent_change else computed * 100
                expression = f"{_display_number(left)} / {_display_number(right)}"
                if is_percent_change:
                    expression += " - 1"
                finding = _arithmetic_finding(
                    stated, computed, left, right, actual_suffix, expression
                )
            if finding is not None:
                findings.append(finding)
    return findings


_METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "sma50": (
        r"\b50[- ]?day\s+(?:SMA|MA|moving average)\b",
        r"50\s*日\s*(?:SMA|均線|均线|均价|均價|移動平均|移动平均)",
    ),
    "sma200": (
        r"\b200[- ]?day\s+(?:SMA|MA|moving average)\b",
        r"200\s*日\s*(?:SMA|均線|均线|均价|均價|移動平均|移动平均)",
    ),
    "atr": (r"\bATR\b", r"平均真實波幅", r"平均真实波幅"),
    "debt_to_equity": (r"\bdebt[- ]?to[- ]?equity\b", r"債務權益比", r"负债权益比", r"債務股本比"),
    "current_ratio": (r"\bcurrent ratio\b", r"流動比率", r"流动比率"),
    "week52_high": (r"\b52[- ]?week high\b", r"52\s*週\s*高", r"52\s*周\s*高"),
    "week52_low": (r"\b52[- ]?week low\b", r"52\s*週\s*低", r"52\s*周\s*低"),
}

# Metrics that legitimately carry one value per period. They must stay out of
# the conflict check — a report listing five quarters of operating cash flow is
# not contradicting itself — but they are exactly what the cross-label check
# needs, so they are collected separately.
_SERIES_ALIASES: dict[str, tuple[str, ...]] = {
    "ocf": (
        r"\boperating cash flow\b", r"\bOCF\b",
        r"(?:經營|经营|營運|营运)(?:活動|活动)?現金流", r"(?:經營|经营|營運|营运)(?:活动)?现金流",
    ),
    "fcf": (
        r"\bfree cash flow\b", r"\bFCF\b", r"自由現金流", r"自由现金流",
    ),
}

# Pairs that must never carry the same number, and what confusing them means.
_CROSSLABEL_PAIRS: tuple[tuple[str, str, str], ...] = (
    (
        "ocf", "fcf",
        "Operating and free cash flow differ by capital expenditure. The same figure "
        "appearing under both labels means one of them was read out of the wrong column.",
    ),
)
_RATIO_METRICS = {"debt_to_equity", "current_ratio"}

# How far two statements of one metric may diverge before it is a conflict,
# as a fraction of the smaller value.
#
# Moving averages and 52-week extremes get a tight bound on purpose. The defect
# this check exists for — a 50-day average given as 514.33 in one section and
# 512.95 in another, because two vendors computed it on different bases — is a
# 0.27% spread. A 1% bound would sail straight past it. Ratios and EPS get the
# looser bound: they are routinely quoted rounded in prose, and flagging that
# would train the reader to skip the whole block.
_DEFAULT_TOLERANCE = 0.01
_METRIC_TOLERANCE: dict[str, float] = {
    "sma50": 0.0015,
    "sma200": 0.0015,
    "week52_high": 0.0015,
    "week52_low": 0.0015,
}

# Markers that mean "this number is not the metric's value in its own units".
# A price level is quoted in currency, never as a multiple or a percentage:
# "ATR 的 1.5 倍" is a multiplier and "ATR of 8.2%" is ATR expressed as a share
# of price. Counting either as a reading conflicts it with the real 8.32.
_MULTIPLE_MARKERS = frozenset({"x", "X", "倍"})
_NON_PRICE_MARKERS = _MULTIPLE_MARKERS | {"%"}


@dataclass(frozen=True)
class _MetricValue:
    value: float
    display: str
    marker: str | None
    approximate: bool


# What may legitimately sit between a metric's label and its value: table
# pipes, colons, brackets, currency marks, an emphasis marker, a short
# parenthesised qualifier such as "(MRQ)", and a linking word. Anything else
# means the number belongs to a different clause — "debt-to-equity ratio,
# undertaking a $100B capital programme" is prose about capex, not a reading of
# leverage, and reading it as one is how the linter invented a 49-vs-100
# conflict on a report that had none.
_BIND_PUNCT = r"[\s|:：=＝\-–—*_`\"'“”$¥€£~≈()（）\[\]【】]"
_BIND_QUALIFIER = r"(?:\([A-Za-z0-9 .%/]{1,10}\)|（[^）]{1,10}）)"
_BIND_WORD = r"(?:of|is|at|was|are|were|to|stands?|currently|now|reads?|為|为|是|達|达|約|约)"
_BINDER_RE = re.compile(
    rf"^(?:{_BIND_PUNCT}|{_BIND_QUALIFIER}|{_BIND_WORD})*$", re.IGNORECASE
)
# A value never sits this far from its label; beyond it we are reading prose.
_MAX_BIND_CHARS = 24


# An indicator's period sits in brackets right after its name: "ATR (14)",
# "RSI (14)", "MACD (12, 26, 9)". Those are parameters, and reading 14 as an ATR
# of 14 dollars conflicts it with the real 8.09. A value in brackets carries a
# decimal or a currency mark ("SMA ($110.60)"), so requiring bare integers keeps
# the two apart; a genuinely round value is skipped rather than misread, which
# loses a reading instead of inventing a conflict.
_PARAMETER_GROUP_RE = re.compile(r"^\s*\(\s*\d{1,3}(?:\s*,\s*\d{1,3})*\s*\)")


def _skip_parameter_group(nearby: str) -> str:
    """Drop a leading indicator-period group so the value after it is found."""
    match = _PARAMETER_GROUP_RE.match(nearby)
    return nearby[match.end() :] if match else nearby


def _is_bound_to_label(gap: str) -> bool:
    """True when only binder text separates a metric's label from a number."""
    return len(gap) <= _MAX_BIND_CHARS and _BINDER_RE.match(gap) is not None


def _is_part_of_a_metric_name(text: str) -> bool:
    """True when ``text`` starts with a number that is part of a metric's name.

    Metric names embed periods ("200 日均價", "52-week high"), so a naive
    "first number after the alias" read can pick up the *next* metric's label
    instead of this one's value.
    """
    return any(
        re.match(alias, text, flags=re.IGNORECASE)
        for aliases in _METRIC_ALIASES.values()
        for alias in aliases
    )


# Two numbers joined by a slash or by "vs" are a pair whose sides cannot be told
# apart positionally.
_PAIR_JOINERS = (r"/", r"vs\.?", r"versus")
_PAIR_AFTER_RE = re.compile(rf"^(?:{'|'.join(_PAIR_JOINERS)})\s*", re.IGNORECASE)
_PAIR_BEFORE_RE = re.compile(rf"(?:{'|'.join(_PAIR_JOINERS)})\s*$", re.IGNORECASE)


def _is_ambiguous_pair(nearby: str, number_match: re.Match) -> bool:
    """True when the number is one side of a joined pair of numbers.

    "50日/200日均價 | 512.95/313.16" packs two metrics into one cell, and
    "50-day SMA ($99.43 vs $110.60)" compares the price against the average —
    the number nearest the label is the *comparand*, not the metric. Positional
    extraction cannot say which side is which, so the honest move is to use
    neither rather than guess and report a phantom conflict.
    """
    before = nearby[: number_match.start()].rstrip()
    after = nearby[number_match.end() :].lstrip()
    if _PAIR_BEFORE_RE.search(before):
        return True
    joined = _PAIR_AFTER_RE.match(after)
    if joined is None:
        return False
    remainder = after[joined.end() :].lstrip("$¥€£ ")
    return _NUMBER_RE.match(remainder) is not None


def _metric_values(
    markdown: str, table: dict[str, tuple[str, ...]] | None = None
) -> dict[str, list[_MetricValue]]:
    table = _METRIC_ALIASES if table is None else table
    values: dict[str, list[_MetricValue]] = {metric: [] for metric in table}
    for metric, aliases in table.items():
        for alias in aliases:
            for match in re.finditer(alias, markdown, flags=re.IGNORECASE):
                # The next 40 characters are intentionally local to the alias:
                # it supports markdown table cells without letting a later field
                # accidentally become this metric's value.
                # Stop at the end of the line. A metric's value always sits on
                # the same line as its label, in prose and in table rows alike.
                # Reading across a newline made "…在50日均线附近\n2. 高波动性…"
                # contribute the list marker 2 as a 50-day average.
                nearby = markdown[match.end() : match.end() + 40].split("\n", 1)[0]
                nearby = _skip_parameter_group(nearby)
                number_match = _NUMBER_RE.search(nearby)
                if number_match is None:
                    continue
                parsed = _parse_number(number_match.group(1))
                if parsed is None:
                    continue
                parsed = _apply_detached_sign(parsed, nearby[: number_match.start()])
                if not _is_bound_to_label(nearby[: number_match.start()]):
                    # The label is mentioned, but this number belongs to another
                    # clause. Reading it as the metric's value is what produced
                    # phantom conflicts on prose-heavy reports.
                    continue
                if _is_part_of_a_metric_name(nearby[number_match.start() :]):
                    # "，但远高于200日均价" after a 50-day alias: the 200 is the next
                    # metric's name, not this metric's value. Reading it as one
                    # put "200" in the list of 50-day averages.
                    continue
                if _is_ambiguous_pair(nearby, number_match):
                    # A combined cell — "50日/200日均價 | 512.95/313.16" — carries
                    # two metrics' values in one field. Which number belongs to
                    # which is not recoverable positionally, so take neither.
                    continue
                marker_match = re.match(r"\s*(%|[xX]|倍)", nearby[number_match.end() :])
                marker = marker_match.group(1) if marker_match else None
                prefix = nearby[: number_match.start()]
                approximate = bool(
                    re.search(r"約|约|\b(?:about|approximately|approx\.?|around|circa|ca\.)\b|[~≈∼]", prefix, re.IGNORECASE)
                )
                values[metric].append(
                    _MetricValue(parsed, number_match.group(1), marker, approximate)
                )
    return values


def _distinct_values(values: list[_MetricValue]) -> list[_MetricValue]:
    distinct: dict[float, _MetricValue] = {}
    for value in values:
        distinct.setdefault(value.value, value)
    return [distinct[number] for number in sorted(distinct)]


def _metric_findings(markdown: str) -> list[Finding]:
    findings: list[Finding] = []
    for metric, occurrences in _metric_values(markdown).items():
        # A price level is quoted in currency, so a number next to one of these
        # aliases carrying a multiple or percent marker is describing something
        # else ("1.5x ATR", "ATR of 8.2%") rather than reading the metric.
        if metric not in _RATIO_METRICS:
            occurrences = [v for v in occurrences if v.marker not in _NON_PRICE_MARKERS]

        # An explicit hedge ("約 514", "approximately 514") is the writer saying
        # the figure is deliberately rounded. Such a value neither proves a
        # conflict nor excuses one, so it sits out the comparison entirely.
        # Letting it widen the tolerance instead meant one hedged mention
        # anywhere in the report silenced the check for every precise one.
        distinct = _distinct_values([v for v in occurrences if not v.approximate])
        if len(distinct) >= 2:
            smallest, largest = distinct[0].value, distinct[-1].value
            spread = largest - smallest
            tolerance = _METRIC_TOLERANCE.get(metric, _DEFAULT_TOLERANCE)
            # These are normally positive market figures.  Treat a zero base
            # conservatively so it cannot cause division by zero in a failed
            # report, while preserving the relative rule otherwise.
            conflicts = spread > 0 if smallest == 0 else spread > tolerance * abs(smallest)
            if conflicts:
                rendered_values = ", ".join(value.display for value in distinct)
                findings.append(
                    Finding(
                        kind="conflict",
                        summary=(
                            f"{metric} is stated as both {_display_number(smallest)} "
                            f"and {_display_number(largest)}"
                        ),
                        detail=f"Distinct values found for {metric}: {rendered_values}.",
                    )
                )

        if metric not in _RATIO_METRICS:
            continue
        percents = [value for value in occurrences if value.marker == "%"]
        multiples = [value for value in occurrences if value.marker in {"x", "X", "倍"}]
        def uses_same_number(percent: _MetricValue, multiple: _MetricValue) -> bool:
            scale = max(abs(percent.value), abs(multiple.value))
            return percent.value == multiple.value if scale == 0 else abs(percent.value - multiple.value) <= 0.01 * scale

        if any(uses_same_number(percent, multiple) for percent in percents for multiple in multiples):
            percent_values = ", ".join(value.display + "%" for value in percents)
            multiple_values = ", ".join(
                value.display + ("倍" if value.marker == "倍" else "x") for value in multiples
            )
            findings.append(
                Finding(
                    kind="unit",
                    summary=f"{metric} is stated as both a percent and a multiple",
                    detail=(
                        f"Percent form: {percent_values}; multiple form: {multiple_values}. "
                        "The numerically matching values use incompatible units."
                    ),
                )
            )
    return findings


def _table_series_values(markdown: str) -> dict[str, list[_MetricValue]]:
    """Read series values out of markdown tables by column, not by adjacency.

    A table says what a number is through its header, which sits on a different
    line — so the same-line rule that keeps prose extraction honest makes tables
    invisible. That matters here: the column *is* the label, and reading a value
    from the wrong one is the defect this check exists to find.
    """
    found: dict[str, list[_MetricValue]] = {metric: [] for metric in _SERIES_ALIASES}
    rows = markdown.split("\n")
    index = 0
    while index < len(rows):
        if not rows[index].lstrip().startswith("|"):
            index += 1
            continue
        block = []
        while index < len(rows) and rows[index].lstrip().startswith("|"):
            block.append(rows[index])
            index += 1
        if len(block) < 3:
            continue
        header = [cell.strip() for cell in block[0].strip().strip("|").split("|")]
        columns: dict[int, str] = {}
        for position, cell in enumerate(header):
            for metric, aliases in _SERIES_ALIASES.items():
                if any(re.search(alias, cell, flags=re.IGNORECASE) for alias in aliases):
                    columns[position] = metric
                    break
        if not columns:
            continue
        for line in block[2:]:  # skip the |---| separator
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            for position, metric in columns.items():
                if position >= len(cells):
                    continue
                match = _NUMBER_RE.search(cells[position])
                if match is None:
                    continue
                parsed = _parse_number(match.group(1))
                if parsed is None:
                    continue
                parsed = _apply_detached_sign(parsed, cells[position][: match.start()])
                found[metric].append(_MetricValue(parsed, match.group(1), None, False))
    return found


def _crosslabel_findings(markdown: str) -> list[Finding]:
    """Flag one number carried under two labels that cannot share a value.

    The 2026-08-06 INTC report tabled Q1 correctly — operating cash flow 1.10B,
    free cash flow -2.54B — and then the bear quoted -2.54B as *operating* cash
    flow, the research manager repeated it, and the portfolio manager published
    it as a "verified" figure. It was verified; it was simply the wrong column.
    Neither the arithmetic check nor the conflict check can see this, because no
    single number is wrong — the error is which name it was given.
    """
    findings: list[Finding] = []
    values = _metric_values(markdown, _SERIES_ALIASES)
    for metric, extra in _table_series_values(markdown).items():
        values.setdefault(metric, []).extend(extra)
    for left, right, explanation in _CROSSLABEL_PAIRS:
        left_values = [v for v in values.get(left, []) if v.value]
        right_values = [v for v in values.get(right, []) if v.value]
        shared: dict[float, tuple[_MetricValue, _MetricValue]] = {}
        for a in left_values:
            for b in right_values:
                scale = max(abs(a.value), abs(b.value))
                if scale and abs(a.value - b.value) <= 0.005 * scale:
                    shared.setdefault(a.value, (a, b))
        for value, (a, _b) in sorted(shared.items()):
            findings.append(
                Finding(
                    kind="crosslabel",
                    summary=(
                        f"{_display_number(value)} appears as both {left.upper()} "
                        f"and {right.upper()}"
                    ),
                    detail=(
                        f"The value {a.display} is labelled {left.upper()} in one place and "
                        f"{right.upper()} in another. {explanation}"
                    ),
                )
            )
    return findings


_WARNING_HEADING = "## ⚠️ Numeric consistency warnings"


def _strip_existing_warnings(markdown: str) -> str:
    """Remove a warning block this module previously inserted.

    In the normal write path the lint runs on the assembled body, before any
    block exists. But re-linting a saved report feeds the block's own evidence
    line — "Distinct values found for atr: 8, 8.2, 8.32." — back in as three
    more readings of ATR, so the linter reports a conflict it created itself.
    """
    if _WARNING_HEADING not in markdown:
        return markdown
    kept, skipping = [], False
    for line in markdown.split("\n"):
        if _WARNING_HEADING in line and line.lstrip().startswith(">"):
            skipping = True
            continue
        if skipping:
            if line.lstrip().startswith(">") or not line.strip():
                continue
            skipping = False
        kept.append(line)
    return "\n".join(kept)


def lint_report(markdown: str) -> list[Finding]:
    """Return numeric findings without ever allowing linting to fail a report save."""
    try:
        if not isinstance(markdown, str):
            return []
        markdown = _strip_existing_warnings(markdown)
        return (
            _division_findings(markdown)
            + _metric_findings(markdown)
            + _crosslabel_findings(markdown)
        )
    except Exception:
        return []


def render_warning_block(findings: list[Finding]) -> str:
    """Render findings as the prominent warning block for the consolidated report."""
    if not findings:
        return ""
    severity = {"arithmetic": 0, "unit": 1, "crosslabel": 2, "conflict": 3}
    ordered = sorted(findings, key=lambda finding: severity.get(finding.kind, len(severity)))
    lines = [
        "> ## ⚠️ Numeric consistency warnings",
        ">",
        "> These were found by a deterministic pass over the finished report. Each one is",
        "> a figure that does not reconcile with other figures in the same report. Verify",
        "> before acting on any conclusion that depends on them.",
    ]
    for finding in ordered:
        lines.extend(
            [
                ">",
                f"> **[{finding.kind}] {finding.summary}**",
                f"> {finding.detail}",
            ]
        )
    return "\n".join(lines)
