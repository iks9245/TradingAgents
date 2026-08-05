"""Render a run's markdown report tree as one browsable HTML page.

``write_report_tree`` already produces good markdown; this reads it back and
makes it comfortable to read in a browser — section navigation, styled tables,
and both themes — without changing what the pipeline writes. Markdown stays the
source of truth, so an HTML page can always be regenerated from an archived run
and never becomes the only copy.

Input is either a report directory (the one ``save_reports`` returns) or a
``full_states_log_*.json`` from ``results_dir``, since runs before this existed
have the JSON but may not have kept the tree.
"""

from __future__ import annotations

import json
import re
from html import escape
from pathlib import Path

from markdown_it import MarkdownIt

from .theme import render_page

# Section order mirrors write_report_tree's numbering, so a rendered page reads
# in the same order as the pipeline ran.
_JSON_SECTIONS = (
    ("market_report", "Market Analyst"),
    ("sentiment_report", "Sentiment Analyst"),
    ("news_report", "News Analyst"),
    ("fundamentals_report", "Fundamentals Analyst"),
    ("investment_plan", "Research Manager"),
    ("trader_investment_decision", "Trader"),
    ("final_trade_decision", "Portfolio Manager"),
)

_HEADING_RE = re.compile(r"^(#{2,3})\s+(.*)$", re.MULTILINE)


def _markdown() -> MarkdownIt:
    # "commonmark" plus tables: the reports lean on markdown tables heavily
    # (every analyst prompt asks for a summary table), and they are the main
    # thing that reads badly as plain text.
    md = MarkdownIt("commonmark", {"html": False, "linkify": True})
    md.enable("table")
    md.enable("strikethrough")
    return md


def _slug(text: str, used: set[str]) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "section"
    slug = base
    n = 2
    while slug in used:
        slug = f"{base}-{n}"
        n += 1
    used.add(slug)
    return slug


def _anchor_headings(html: str, headings: list[tuple[str, str, int]]) -> str:
    """Give each rendered h2/h3 an id matching the table of contents."""
    queue = list(headings)

    def replace(match):
        if not queue:
            return match.group(0)
        slug, _text, _level = queue.pop(0)
        return f"<{match.group(1)} id=\"{slug}\">"

    return re.sub(r"<(h[23])>", replace, html)


def _collect_headings(markdown: str) -> list[tuple[str, str, int]]:
    used: set[str] = set()
    out = []
    for match in _HEADING_RE.finditer(markdown):
        level = len(match.group(1))
        text = match.group(2).strip()
        out.append((_slug(text, used), text, level))
    return out


def _table_of_contents(headings: list[tuple[str, str, int]]) -> str:
    if len(headings) < 2:
        return ""
    items = []
    for slug, text, level in headings:
        # Unordered, not numbered. Analyst-authored reports carry their own
        # headings, so the list interleaves structural sections with content
        # ones; numbering them would imply a flat sequence that does not exist.
        # Indentation carries the level instead.
        indent = ' style="margin-left:1rem"' if level == 3 else ""
        items.append(f'<li{indent}><a href="#{slug}">{escape(text)}</a></li>')
    return (
        '<nav class="toc" aria-label="Sections"><strong>Sections</strong>'
        f'<ul>{"".join(items)}</ul></nav>'
    )


def markdown_to_page(markdown: str, *, title: str, subtitle: str = "") -> str:
    """Render arbitrary report markdown as a full themed HTML page."""
    headings = _collect_headings(markdown)
    body_html = _anchor_headings(_markdown().render(markdown), headings)
    # Tables are the widest thing in these reports; let each scroll in its own
    # box rather than making the page scroll sideways on a phone.
    body_html = body_html.replace("<table>", '<div class="scroll-x"><table>').replace(
        "</table>", "</table></div>"
    )
    return render_page(
        title,
        _table_of_contents(headings) + f'<article class="report">{body_html}</article>',
        subtitle=subtitle,
    )


def render_report_dir(report_dir: str | Path, *, title: str | None = None) -> str:
    """Render a directory written by ``write_report_tree``.

    Prefers ``complete_report.md``; falls back to concatenating the per-section
    files in their numbered order when only those are present.
    """
    report_dir = Path(report_dir).expanduser()
    if not report_dir.is_dir():
        raise FileNotFoundError(f"not a directory: {report_dir}")

    complete = report_dir / "complete_report.md"
    if complete.exists():
        markdown = complete.read_text(encoding="utf-8")
    else:
        parts = []
        for path in sorted(report_dir.rglob("*.md")):
            parts.append(f"## {path.stem.replace('_', ' ').title()}\n\n" + path.read_text(encoding="utf-8"))
        if not parts:
            raise FileNotFoundError(f"no markdown reports under {report_dir}")
        markdown = "\n\n".join(parts)

    return markdown_to_page(
        markdown,
        title=title or f"Analysis Report — {report_dir.name}",
        subtitle=f"Rendered from {escape(str(report_dir))}",
    )


def render_state_log(log_path: str | Path, *, title: str | None = None) -> str:
    """Render a ``full_states_log_*.json`` written by ``_log_state``.

    Older runs kept only this file. Debate transcripts are included so the
    reasoning is inspectable, not just the conclusion.
    """
    log_path = Path(log_path).expanduser()
    state = json.loads(log_path.read_text(encoding="utf-8"))

    ticker = state.get("company_of_interest", "unknown")
    trade_date = state.get("trade_date", "unknown")

    parts = []
    for key, label in _JSON_SECTIONS:
        content = state.get(key)
        if content:
            parts.append(f"## {label}\n\n{content}")

    debate = state.get("investment_debate_state") or {}
    if debate.get("history"):
        parts.append(f"## Research Debate\n\n{debate['history']}")
    risk = state.get("risk_debate_state") or {}
    if risk.get("history"):
        parts.append(f"## Risk Debate\n\n{risk['history']}")

    if not parts:
        raise ValueError(f"no report content found in {log_path}")

    return markdown_to_page(
        "\n\n".join(parts),
        title=title or f"{ticker} — {trade_date}",
        subtitle=f"Rendered from {escape(log_path.name)}",
    )


def render_any(source: str | Path, *, title: str | None = None) -> str:
    """Render whichever of the two supported inputs ``source`` happens to be."""
    path = Path(source).expanduser()
    if path.is_dir():
        return render_report_dir(path, title=title)
    if path.suffix == ".json":
        return render_state_log(path, title=title)
    if path.suffix in (".md", ".markdown"):
        return markdown_to_page(
            path.read_text(encoding="utf-8"),
            title=title or path.stem,
            subtitle=f"Rendered from {escape(path.name)}",
        )
    raise ValueError(
        f"don't know how to render {path}: expected a report directory, a "
        "full_states_log JSON, or a markdown file"
    )
