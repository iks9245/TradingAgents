"""Index saved analysis runs so failures and provenance stay visible at a glance.

Reports are useful long after their generating process has exited, but their
directory tree makes incomplete runs, warnings, and unknown revisions easy to
miss.  This module extracts that durable on-disk evidence without rerunning any
analysis, and renders it in formats that remain useful outside the application.
"""

from __future__ import annotations

import argparse
import html
import re
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from tradingagents.agents.utils.evidence_policy import UNVALIDATED_MARKER
from tradingagents.agents.utils.rating import parse_rating, parse_trader_action
from tradingagents.webreport.theme import (
    RATING_SCALE_DARK,
    RATING_SCALE_LIGHT,
    render_page,
)

_RUN_ID_RE = re.compile(r"^([A-Z0-9.\-^]+)_(\d{8})_(\d{6})$")
_WARNING_RE = re.compile(r"^> \*\*\[(\w+)\] ", re.MULTILINE)
_PRICE_TARGET_RE = re.compile(
    r"^\s*\*{0,2}Price Target\*{0,2}\s*:\s*(.*?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _plural(count: int, noun: str) -> str:
    """Count and noun, agreeing. Both are read by people, so both should."""
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


@dataclass(frozen=True)
class RunRecord:
    """Metadata preserved by one report directory."""

    run_id: str
    ticker: str
    started_at: datetime | None
    path: Path
    generated_at: str | None
    code_revision: str | None
    rating: str | None
    price_target: float | None
    action: str | None
    warnings: dict[str, int]
    unvalidated_sections: int
    complete: bool

    @property
    def warning_count(self) -> int:
        return sum(self.warnings.values())

    @property
    def statuses(self) -> tuple[str, ...]:
        labels: list[str] = []
        if not self.complete:
            labels.append("incomplete")
        if self.warning_count:
            labels.append(_plural(self.warning_count, "warning"))
        if self.unvalidated_sections:
            labels.append(f"{self.unvalidated_sections} unvalidated")
        if self.code_revision is None:
            labels.append("unknown code")
        elif self.code_revision.endswith("+local-changes"):
            labels.append("local changes")
        return tuple(labels or ["ok"])


def _read_text(path: Path) -> str:
    """Return an empty value when a run artifact cannot be read.

    A scan is an observability path: one damaged file must not hide that run or
    prevent healthy siblings from appearing in the index.
    """
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return ""


def _parse_header(report: str) -> tuple[str | None, str | None]:
    lines = report.splitlines()
    if len(lines) < 3 or not lines[2].startswith("Generated:"):
        return None, None

    generated, separator, revision = lines[2].removeprefix("Generated:").partition(
        " · code "
    )
    generated_at = generated.strip() or None
    if not separator:
        return generated_at, None

    revision = revision.strip()
    if not revision or revision == "unknown" or revision.startswith("unknown —"):
        return generated_at, None
    return generated_at, revision


def _parse_price_target(text: str) -> float | None:
    match = _PRICE_TARGET_RE.search(text)
    if match is None:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _scan_run(path: Path, match: re.Match[str]) -> RunRecord:
    ticker, date_text, time_text = match.groups()
    try:
        started_at = datetime.strptime(date_text + time_text, "%Y%m%d%H%M%S")
    except ValueError:
        started_at = None

    report_path = path / "complete_report.md"
    report = _read_text(report_path)
    generated_at, code_revision = _parse_header(report)

    decision_path = path / "5_portfolio" / "decision.md"
    # An empty or unreadable decision is itself malformed evidence. Only a
    # genuinely absent artifact should fall back to the consolidated report.
    decision = _read_text(decision_path) if decision_path.is_file() else report
    rating = parse_rating(decision, default="") or None

    trader_path = path / "3_trading" / "trader.md"
    trader = _read_text(trader_path) if trader_path.is_file() else report
    action = parse_trader_action(trader)

    warnings = dict(Counter(_WARNING_RE.findall(_read_text(path / "numeric_warnings.md"))))
    return RunRecord(
        run_id=path.name,
        ticker=ticker,
        started_at=started_at,
        path=path,
        generated_at=generated_at,
        code_revision=code_revision,
        rating=rating,
        price_target=_parse_price_target(decision),
        action=action,
        warnings=warnings,
        unvalidated_sections=report.count(UNVALIDATED_MARKER),
        complete=bool(report),
    )


def is_run_directory(path: Path | str) -> bool:
    """True when ``path`` is named like a run :func:`scan_runs` would recognize.

    Exposed so a writer can ask whether the directory it just filled belongs to
    an index, rather than assuming every save path is one.
    """
    return _RUN_ID_RE.fullmatch(Path(path).name) is not None


def scan_runs(reports_dir: Path | str = "reports") -> list[RunRecord]:
    """Return recognized report runs newest-first, tolerating damaged artifacts."""
    root = Path(reports_dir).expanduser()
    try:
        children = list(root.iterdir())
    except OSError:
        return []

    records: list[RunRecord] = []
    for path in children:
        match = _RUN_ID_RE.fullmatch(path.name)
        if match is None or not path.is_dir():
            continue
        records.append(_scan_run(path, match))

    # The run timestamp is authoritative. The run ID is a deterministic tie
    # breaker, while malformed timestamp suffixes remain visible at the end.
    records.sort(
        key=lambda record: (
            record.started_at is not None,
            record.started_at or datetime.min,
            record.run_id,
        ),
        reverse=True,
    )
    return records


def _started_text(record: RunRecord) -> str:
    if record.started_at is None:
        return "—"
    return record.started_at.strftime("%Y-%m-%d %H:%M:%S")


def _target_text(record: RunRecord) -> str:
    return "—" if record.price_target is None else f"{record.price_target:g}"


def _markdown_cell(value: object) -> str:
    return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def render_index_markdown(records: Sequence[RunRecord]) -> str:
    """Render a deterministic GitHub-flavoured summary table."""
    lines = [
        "| Ticker | Started | Rating | Target | Code | Status |",
        "| --- | --- | --- | ---: | --- | --- |",
    ]
    for record in records:
        values = (
            record.ticker,
            _started_text(record),
            record.rating or "—",
            _target_text(record),
            record.code_revision or "—",
            ", ".join(record.statuses),
        )
        lines.append("| " + " | ".join(_markdown_cell(value) for value in values) + " |")
    return "\n".join(lines) + "\n"


def _rating_style(rating: str | None) -> str:
    if rating not in RATING_SCALE_LIGHT or rating not in RATING_SCALE_DARK:
        return ""
    token = rating.lower()
    return f' style="color: var(--rating-{token}); font-weight: 600"'


def render_index_html(records: Sequence[RunRecord]) -> str:
    """Render a self-contained report index using the shared report shell."""
    rows: list[str] = []
    for record in records:
        statuses = record.statuses
        if statuses == ("ok",):
            row_attributes = ""
        else:
            row_attributes = (
                ' class="run-needs-attention" '
                'style="background: var(--surface); border-left: 3px solid var(--negative)"'
            )

        report_name = (
            "complete_report.html"
            if (record.path / "complete_report.html").is_file()
            else "complete_report.md"
        )
        href = html.escape(f"{record.run_id}/{report_name}", quote=True)
        ticker = html.escape(record.ticker)
        started = html.escape(_started_text(record))
        rating = html.escape(record.rating or "—")
        target = html.escape(_target_text(record))
        revision = html.escape(record.code_revision or "—")
        status = html.escape(", ".join(statuses))
        rows.append(
            f"<tr{row_attributes}>"
            f'<td><a href="{href}">{ticker}</a></td>'
            f"<td>{started}</td>"
            f"<td{_rating_style(record.rating)}>{rating}</td>"
            f'<td class="num">{target}</td>'
            f"<td><code>{revision}</code></td>"
            f"<td>{status}</td>"
            "</tr>"
        )

    body = (
        '<table class="run-index">'
        "<thead><tr>"
        "<th>Ticker</th><th>Started</th><th>Rating</th>"
        '<th class="num">Target</th><th>Code</th><th>Status</th>'
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
    )
    return render_page(
        title="Analysis runs",
        body=body,
        subtitle=_plural(len(records), "run"),
    )


def write_index(
    reports_dir: Path | str, *, out: Path | str | None = None, markdown: bool = False
) -> Path:
    """Scan ``reports_dir`` and write its index; return the path written.

    Raises ``OSError`` if the file cannot be written. Scanning itself never
    raises, so a damaged run still reaches the page.
    """
    reports_dir = Path(reports_dir).expanduser()
    records = scan_runs(reports_dir)
    content = render_index_markdown(records) if markdown else render_index_html(records)

    default_name = "index.md" if markdown else "index.html"
    destination = Path(out).expanduser() if out else reports_dir / default_name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8")
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tradingagents.run_index",
        description="Build an index of saved TradingAgents analysis runs.",
    )
    parser.add_argument(
        "reports_dir",
        nargs="?",
        default="reports",
        help="reports directory (default: reports)",
    )
    parser.add_argument(
        "-o",
        "--out",
        help="output path (default: REPORTS_DIR/index.html or index.md)",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="write a Markdown table instead of a self-contained HTML page",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        out = write_index(args.reports_dir, out=args.out, markdown=args.markdown)
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
