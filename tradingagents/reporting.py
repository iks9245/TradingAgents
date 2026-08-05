"""Reusable report-tree writer shared by the CLI and the programmatic API.

Writes a run's per-section markdown (analysts, research, trading, risk,
portfolio) plus a consolidated ``complete_report.md`` under ``save_path``. The
CLI and ``TradingAgentsGraph.save_reports`` both call this, so a headless / API
run produces the same on-disk report tree a CLI run does.

A browsable ``complete_report.html`` is written alongside by default. Markdown
remains the source of truth: the HTML is generated *from* it, and if generation
fails the markdown tree is already on disk and is kept. Losing a finished run's
report because a renderer raised would be far worse than not having the HTML.
Set ``report_html`` to false in config (or ``TRADINGAGENTS_REPORT_HTML=false``)
to skip it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReportPaths:
    """Where a run's reports landed.

    ``html`` is None when the HTML was disabled or could not be generated, so
    callers can report only what actually exists rather than printing a path to
    a file that is not there.
    """

    markdown: Path
    html: Path | None = None


def _html_enabled(explicit: bool | None) -> bool:
    """Resolve the HTML setting: explicit argument wins, else live config."""
    if explicit is not None:
        return explicit
    try:
        from tradingagents.dataflows.config import get_config

        return bool(get_config().get("report_html", True))
    except Exception:  # config not initialised (bare library use)
        return True


def write_report_tree(final_state: dict, ticker: str, save_path, *, html: bool | None = None) -> Path:
    """Save a completed run's reports to ``save_path``; return the complete-report path.

    Kept returning the markdown path for backwards compatibility. Use
    :func:`write_report_bundle` when you also need the HTML path.
    """
    return write_report_bundle(final_state, ticker, save_path, html=html).markdown


def write_report_bundle(
    final_state: dict, ticker: str, save_path, *, html: bool | None = None
) -> ReportPaths:
    """Write the markdown tree and, unless disabled, the HTML page beside it."""
    markdown_path = _write_markdown_tree(final_state, ticker, save_path)

    html_path = None
    if _html_enabled(html):
        html_path = _write_html(markdown_path, ticker)

    return ReportPaths(markdown=markdown_path, html=html_path)


def _write_html(markdown_path: Path, ticker: str) -> Path | None:
    """Render the consolidated markdown to a sibling HTML page.

    Failures are logged and swallowed on purpose — the markdown tree is already
    written by this point, and a rendering bug must not turn a completed
    analysis into a failed save.
    """
    try:
        from tradingagents.webreport import markdown_to_page

        markdown = markdown_path.read_text(encoding="utf-8")
        body, generated = _split_own_header(markdown)
        page = markdown_to_page(
            body,
            title=f"Trading Analysis Report: {ticker}",
            subtitle=generated or f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        )
        html_path = markdown_path.with_suffix(".html")
        html_path.write_text(page, encoding="utf-8")
        return html_path
    except Exception as exc:
        logger.warning(
            "Could not write the HTML report (the markdown report at %s is unaffected): %s",
            markdown_path, exc,
        )
        return None


def _split_own_header(markdown: str) -> tuple[str, str]:
    """Peel off the ``# title`` / ``Generated:`` header this module writes.

    The page shell already renders a title and subtitle, so leaving the
    markdown's own header in the body prints both twice. Only the exact header
    written by :func:`_write_markdown_tree` is removed — anything else is left
    alone, so this cannot silently eat a report's first section.

    Returns ``(body, generated_line)``; ``generated_line`` is empty when no
    matching header was found.
    """
    lines = markdown.split("\n")
    if not lines or not lines[0].startswith("# Trading Analysis Report:"):
        return markdown, ""

    generated = ""
    index = 1
    while index < len(lines) and not lines[index].strip():
        index += 1
    if index < len(lines) and lines[index].startswith("Generated:"):
        generated = lines[index].strip()
        index += 1

    return "\n".join(lines[index:]).lstrip("\n"), generated


def _write_markdown_tree(final_state: dict, ticker: str, save_path) -> Path:
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    sections = []

    # 1. Analysts
    analysts_dir = save_path / "1_analysts"
    analyst_parts = []
    if final_state.get("market_report"):
        analysts_dir.mkdir(exist_ok=True)
        (analysts_dir / "market.md").write_text(final_state["market_report"], encoding="utf-8")
        analyst_parts.append(("Market Analyst", final_state["market_report"]))
    if final_state.get("sentiment_report"):
        analysts_dir.mkdir(exist_ok=True)
        (analysts_dir / "sentiment.md").write_text(final_state["sentiment_report"], encoding="utf-8")
        analyst_parts.append(("Sentiment Analyst", final_state["sentiment_report"]))
    if final_state.get("news_report"):
        analysts_dir.mkdir(exist_ok=True)
        (analysts_dir / "news.md").write_text(final_state["news_report"], encoding="utf-8")
        analyst_parts.append(("News Analyst", final_state["news_report"]))
    if final_state.get("fundamentals_report"):
        analysts_dir.mkdir(exist_ok=True)
        (analysts_dir / "fundamentals.md").write_text(final_state["fundamentals_report"], encoding="utf-8")
        analyst_parts.append(("Fundamentals Analyst", final_state["fundamentals_report"]))
    if analyst_parts:
        content = "\n\n".join(f"### {name}\n{text}" for name, text in analyst_parts)
        sections.append(f"## I. Analyst Team Reports\n\n{content}")

    # 2. Research
    if final_state.get("investment_debate_state"):
        research_dir = save_path / "2_research"
        debate = final_state["investment_debate_state"]
        research_parts = []
        if debate.get("bull_history"):
            research_dir.mkdir(exist_ok=True)
            (research_dir / "bull.md").write_text(debate["bull_history"], encoding="utf-8")
            research_parts.append(("Bull Researcher", debate["bull_history"]))
        if debate.get("bear_history"):
            research_dir.mkdir(exist_ok=True)
            (research_dir / "bear.md").write_text(debate["bear_history"], encoding="utf-8")
            research_parts.append(("Bear Researcher", debate["bear_history"]))
        if debate.get("judge_decision"):
            research_dir.mkdir(exist_ok=True)
            (research_dir / "manager.md").write_text(debate["judge_decision"], encoding="utf-8")
            research_parts.append(("Research Manager", debate["judge_decision"]))
        if research_parts:
            content = "\n\n".join(f"### {name}\n{text}" for name, text in research_parts)
            sections.append(f"## II. Research Team Decision\n\n{content}")

    # 3. Trading
    if final_state.get("trader_investment_plan"):
        trading_dir = save_path / "3_trading"
        trading_dir.mkdir(exist_ok=True)
        (trading_dir / "trader.md").write_text(final_state["trader_investment_plan"], encoding="utf-8")
        sections.append(f"## III. Trading Team Plan\n\n### Trader\n{final_state['trader_investment_plan']}")

    # 4. Risk Management
    if final_state.get("risk_debate_state"):
        risk_dir = save_path / "4_risk"
        risk = final_state["risk_debate_state"]
        risk_parts = []
        if risk.get("aggressive_history"):
            risk_dir.mkdir(exist_ok=True)
            (risk_dir / "aggressive.md").write_text(risk["aggressive_history"], encoding="utf-8")
            risk_parts.append(("Aggressive Analyst", risk["aggressive_history"]))
        if risk.get("conservative_history"):
            risk_dir.mkdir(exist_ok=True)
            (risk_dir / "conservative.md").write_text(risk["conservative_history"], encoding="utf-8")
            risk_parts.append(("Conservative Analyst", risk["conservative_history"]))
        if risk.get("neutral_history"):
            risk_dir.mkdir(exist_ok=True)
            (risk_dir / "neutral.md").write_text(risk["neutral_history"], encoding="utf-8")
            risk_parts.append(("Neutral Analyst", risk["neutral_history"]))
        if risk_parts:
            content = "\n\n".join(f"### {name}\n{text}" for name, text in risk_parts)
            sections.append(f"## IV. Risk Management Team Decision\n\n{content}")

        # 5. Portfolio Manager
        if risk.get("judge_decision"):
            portfolio_dir = save_path / "5_portfolio"
            portfolio_dir.mkdir(exist_ok=True)
            (portfolio_dir / "decision.md").write_text(risk["judge_decision"], encoding="utf-8")
            sections.append(f"## V. Portfolio Manager Decision\n\n### Portfolio Manager\n{risk['judge_decision']}")

    # Write consolidated report
    header = f"# Trading Analysis Report: {ticker}\n\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    (save_path / "complete_report.md").write_text(header + "\n\n".join(sections), encoding="utf-8")
    return save_path / "complete_report.md"
