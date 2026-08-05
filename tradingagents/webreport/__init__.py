"""Self-contained HTML rendering for analysis reports and backtest results.

Every page is a single file with no external requests — inline CSS, inline SVG,
one small theme-toggle script — so it opens from ``file://``, survives being
emailed as an attachment, and can be archived next to the run it describes.

Two entry points:

    # A finished analysis run, browsable with section navigation
    python -m tradingagents.webreport ~/.tradingagents/logs/NVDA/... -o report.html

    # Backtest and ablation results, charted
    python -m tradingagents.backtest ... --html runs/report.html
    python -m tradingagents.backtest.ablation_cli ... --html runs/ablation.html

Programmatically::

    from tradingagents.webreport import render_backtest_html
    Path("report.html").write_text(render_backtest_html(result), encoding="utf-8")
"""

from .analysis import (
    markdown_to_page,
    render_any,
    render_report_dir,
    render_state_log,
)
from .backtest import render_ablation_html, render_backtest_html
from .charts import cumulative_chart, interval_chart, rating_chart
from .theme import build_css, render_page

__all__ = [
    "build_css",
    "cumulative_chart",
    "interval_chart",
    "markdown_to_page",
    "rating_chart",
    "render_ablation_html",
    "render_any",
    "render_backtest_html",
    "render_page",
    "render_report_dir",
    "render_state_log",
]
