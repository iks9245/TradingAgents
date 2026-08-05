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
_NUMBER_RE = re.compile(rf"(?<![\d,])({_NUMBER})(?![\d,])")


def _parse_number(value: str) -> float | None:
    """Parse the report's supported decimal notation, or return ``None``."""
    if not re.fullmatch(_NUMBER, value):
        return None
    try:
        return float(value.replace(",", ""))
    except (TypeError, ValueError, OverflowError):
        return None


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
    "eps_diluted": (r"\bdiluted EPS\b", r"稀釋每股(?:盈餘|收益)", r"每股收益（稀釋）"),
}
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

# Markers that mean "this number is a multiplier, not the metric's value".
# Without this, "ATR 的 1.5 倍" contributes 1.5 as an ATR reading and conflicts
# with the real 40.39 — a false positive on correct prose.
_MULTIPLE_MARKERS = frozenset({"x", "X", "倍"})


@dataclass(frozen=True)
class _MetricValue:
    value: float
    display: str
    marker: str | None
    approximate: bool


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


def _is_ambiguous_pair(nearby: str, number_match: re.Match) -> bool:
    """True when the number is one side of a slash-joined pair of numbers.

    A report that writes "50日/200日均價 | 512.95/313.16" packs two metrics into
    one cell. Positional extraction cannot say which value is which, so the
    honest move is to use neither rather than to guess and report a phantom
    conflict.
    """
    before = nearby[: number_match.start()].rstrip()
    after = nearby[number_match.end() :].lstrip()
    if before.endswith("/"):
        return True
    return after.startswith("/") and _NUMBER_RE.match(after[1:].lstrip()) is not None


def _metric_values(markdown: str) -> dict[str, list[_MetricValue]]:
    values: dict[str, list[_MetricValue]] = {metric: [] for metric in _METRIC_ALIASES}
    for metric, aliases in _METRIC_ALIASES.items():
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
                number_match = _NUMBER_RE.search(nearby)
                if number_match is None:
                    continue
                parsed = _parse_number(number_match.group(1))
                if parsed is None:
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
                    re.search(r"約|约|\b(?:about|approximately|approx\.?|around)\b|~", prefix, re.IGNORECASE)
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
        # A price level is never written as a multiple, so a number carrying a
        # multiple marker next to one of these aliases is a multiplier in prose
        # ("1.5x ATR"), not a reading of the metric.
        if metric not in _RATIO_METRICS:
            occurrences = [v for v in occurrences if v.marker not in _MULTIPLE_MARKERS]

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


def lint_report(markdown: str) -> list[Finding]:
    """Return numeric findings without ever allowing linting to fail a report save."""
    try:
        if not isinstance(markdown, str):
            return []
        return _division_findings(markdown) + _metric_findings(markdown)
    except Exception:
        return []


def render_warning_block(findings: list[Finding]) -> str:
    """Render findings as the prominent warning block for the consolidated report."""
    if not findings:
        return ""
    severity = {"arithmetic": 0, "unit": 1, "conflict": 2}
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
