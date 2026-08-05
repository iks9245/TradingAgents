"""HTML report rendering: self-containment, chart geometry, and escaping.

Everything here is offline. The rendered pages are checked as strings rather
than in a browser, so these tests guard the properties a screenshot would not
catch reliably — no external requests, no unescaped user content, no clipped
labels — while the visual pass stays a manual step.
"""

import json
import re

import pytest

from tradingagents.backtest.grid import EvalPoint, build_grid
from tradingagents.backtest.harness import Backtest
from tradingagents.backtest.metrics import Decision, PairedComparison
from tradingagents.backtest.prices import ForwardReturn, PriceCache
from tradingagents.backtest.strategies import AlwaysRating, UniformRandomRating
from tradingagents.webreport import (
    markdown_to_page,
    render_backtest_html,
    render_page,
    render_state_log,
)
from tradingagents.webreport.analysis import render_any, render_report_dir
from tradingagents.webreport.charts import (
    _axis_ticks,
    _spread_labels,
    _text_width,
    cumulative_chart,
    interval_chart,
    rating_chart,
)
from tradingagents.webreport.theme import build_css

# Anything that would make the browser reach off-page. data: and #fragment are
# fine; a scheme or a path is not.
_EXTERNAL_REF = re.compile(r'(?:src|href)\s*=\s*"(?!#)(?!data:)[^"]+"')


# --- self-containment ---------------------------------------------------


@pytest.mark.unit
def test_page_makes_no_external_requests():
    """Reports open from file:// and get emailed; a CDN reference breaks both."""
    page = render_page("T", "<p>body</p>", subtitle="s")
    assert not _EXTERNAL_REF.search(page)
    assert "<style>" in page and "</style>" in page


@pytest.mark.unit
def test_dark_theme_is_declared_under_both_scopes():
    """The OS media query covers the system setting; data-theme covers the toggle."""
    css = build_css()
    assert "@media (prefers-color-scheme: dark)" in css
    assert ':root[data-theme="dark"]' in css
    # The guard that lets an explicit light stamp beat OS-dark.
    assert ':not([data-theme="light"])' in css


@pytest.mark.unit
def test_page_declares_charset_and_viewport():
    page = render_page("T", "<p>x</p>")
    assert '<meta charset="utf-8">' in page
    assert "width=device-width" in page


# --- axis and label geometry --------------------------------------------


@pytest.mark.unit
def test_axis_ticks_land_on_zero():
    """An appended zero sits at an arbitrary offset and collides with neighbours."""
    ticks = _axis_ticks(-0.0091, 0.0084)
    assert any(abs(t) < 1e-12 for t in ticks)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("lo", "hi"),
    [(-0.009, 0.008), (-0.001, 0.05), (0.0, 0.02), (-0.3, 0.3), (-1e-4, 1e-4)],
)
def test_axis_ticks_are_evenly_spaced_and_unique(lo, hi):
    ticks = _axis_ticks(lo, hi)
    assert len(ticks) == len(set(ticks))
    if len(ticks) > 2:
        gaps = [b - a for a, b in zip(ticks, ticks[1:], strict=False)]
        assert max(gaps) - min(gaps) < max(gaps) * 1e-6


@pytest.mark.unit
def test_spread_labels_enforces_a_minimum_gap():
    """Line ends converge exactly when the result is null — the labels must not."""
    spread = _spread_labels([100.0, 101.0, 102.0], min_gap=15)
    ordered = sorted(spread)
    gaps = [b - a for a, b in zip(ordered, ordered[1:], strict=False)]
    assert min(gaps) >= 15 - 1e-9


@pytest.mark.unit
def test_spread_labels_preserves_order():
    spread = _spread_labels([50.0, 10.0, 30.0], min_gap=20)
    assert spread[1] < spread[2] < spread[0]


@pytest.mark.unit
def test_long_row_labels_widen_the_gutter_instead_of_clipping():
    """SVG clips silently, so a long name would read as a shorter different one."""
    short = interval_chart([("ab", 0.01, -0.01, 0.03, False)])
    long = interval_chart([("a" * 60, 0.01, -0.01, 0.03, False)])
    assert _text_width("a" * 60, 12.5) > _text_width("ab", 12.5)
    # The label anchor moves right as the gutter grows.
    short_x = float(re.search(r'class="row-label" x="([\d.]+)"', short).group(1))
    long_x = float(re.search(r'class="row-label" x="([\d.]+)"', long).group(1))
    assert long_x > short_x


@pytest.mark.unit
def test_series_labels_stay_inside_the_viewbox():
    series = [("a_very_long_strategy_name_indeed", [("2025-01-01", 0.0), ("2025-02-01", 0.01)])]
    svg = cumulative_chart(series)
    view_width = float(re.search(r'viewBox="0 0 ([\d.]+)', svg).group(1))
    label_x = float(re.search(r'class="value-label" x="([\d.]+)"', svg).group(1))
    assert label_x + _text_width("a_very_long_strategy_name_indeed", 12) <= view_width


# --- escaping -----------------------------------------------------------


@pytest.mark.unit
def test_chart_labels_are_escaped():
    """Tickers reach here from LLM tool calls, so they are not trusted input."""
    svg = interval_chart([('<script>alert(1)</script>', 0.0, -0.01, 0.01, False)])
    assert "<script>" not in svg
    assert "&lt;script&gt;" in svg


@pytest.mark.unit
def test_rating_chart_escapes_labels():
    svg = rating_chart([('<img src=x>', {"Buy": 3, "Sell": 1})])
    assert "<img" not in svg
    assert "&lt;img" in svg


# --- empty input --------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("fn", [interval_chart, cumulative_chart, rating_chart])
def test_charts_return_empty_string_for_no_data(fn):
    assert fn([]) == ""


@pytest.mark.unit
def test_rating_chart_skips_rows_with_no_decisions():
    assert rating_chart([("empty", {})]) == ""


# --- markdown rendering -------------------------------------------------


@pytest.mark.unit
def test_markdown_page_renders_tables_and_anchors_headings():
    md = "## Section One\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n### Sub\n\ntext\n"
    page = markdown_to_page(md, title="T")
    assert "<table>" in page
    assert 'id="section-one"' in page
    assert 'href="#section-one"' in page
    # Wide tables scroll in their own box rather than the page going sideways.
    assert '<div class="scroll-x"><table>' in page


@pytest.mark.unit
def test_duplicate_headings_get_unique_anchors():
    page = markdown_to_page("## Notes\n\na\n\n## Notes\n\nb\n", title="T")
    assert 'id="notes"' in page
    assert 'id="notes-2"' in page


@pytest.mark.unit
def test_raw_html_in_report_markdown_is_not_executed():
    """Report bodies are LLM output; treating them as trusted HTML would be a hole."""
    page = markdown_to_page("## S\n\n<script>alert(1)</script>\n", title="T")
    assert "<script>alert(1)</script>" not in page


@pytest.mark.unit
def test_single_heading_gets_no_table_of_contents():
    assert 'class="toc"' not in markdown_to_page("## Only\n\ntext\n", title="T")


# --- file inputs --------------------------------------------------------


@pytest.mark.unit
def test_render_report_dir_prefers_the_complete_report(tmp_path):
    (tmp_path / "complete_report.md").write_text("## Complete\n\nbody\n", encoding="utf-8")
    (tmp_path / "1_analysts").mkdir()
    (tmp_path / "1_analysts" / "market.md").write_text("ignored", encoding="utf-8")
    page = render_report_dir(tmp_path)
    assert "Complete" in page
    assert "ignored" not in page


@pytest.mark.unit
def test_render_report_dir_falls_back_to_section_files(tmp_path):
    (tmp_path / "1_analysts").mkdir()
    (tmp_path / "1_analysts" / "market.md").write_text("market body", encoding="utf-8")
    assert "market body" in render_report_dir(tmp_path)


@pytest.mark.unit
def test_render_state_log_includes_reports_and_debates(tmp_path):
    """Older runs kept only the JSON, and the debates are the interesting part."""
    log = tmp_path / "full_states_log_2026-01-02.json"
    log.write_text(json.dumps({
        "company_of_interest": "NVDA",
        "trade_date": "2026-01-02",
        "market_report": "market body",
        "final_trade_decision": "**Rating**: Hold",
        "investment_debate_state": {"history": "Bull Analyst: up"},
        "risk_debate_state": {"history": "Aggressive: size up"},
    }), encoding="utf-8")

    page = render_state_log(log)
    assert "market body" in page
    assert "Bull Analyst: up" in page
    assert "Aggressive: size up" in page
    assert "NVDA" in page


@pytest.mark.unit
def test_render_state_log_rejects_an_empty_state(tmp_path):
    log = tmp_path / "full_states_log_x.json"
    log.write_text(json.dumps({"company_of_interest": "X"}), encoding="utf-8")
    with pytest.raises(ValueError, match="no report content"):
        render_state_log(log)


@pytest.mark.unit
def test_render_any_rejects_an_unsupported_input(tmp_path):
    path = tmp_path / "thing.txt"
    path.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="don't know how to render"):
        render_any(path)


@pytest.mark.unit
def test_render_any_dispatches_on_type(tmp_path):
    md = tmp_path / "r.md"
    md.write_text("## H\n\nbody\n", encoding="utf-8")
    assert "body" in render_any(md)


# --- backtest page ------------------------------------------------------


@pytest.fixture()
def result():
    import pandas as pd

    grid = build_grid(
        ["AAA", "BBB"], start="2025-01-06", end="2025-10-01",
        holding_days=5, step_days=14, today=pd.Timestamp("2026-06-01").date(),
    )
    prices = PriceCache()
    index = pd.bdate_range(start="2024-12-01", periods=320)
    prices.load_frame("AAA", pd.DataFrame({"Close": [100 * 1.002 ** i for i in range(320)]}, index=index))
    prices.load_frame("BBB", pd.DataFrame({"Close": [100 * 0.998 ** i for i in range(320)]}, index=index))
    prices.load_frame("SPY", pd.DataFrame({"Close": [100.0] * 320}, index=index))
    backtest = Backtest(grid, price_cache=prices, bootstrap_iterations=300)
    return backtest.run(AlwaysRating("Buy", name="sut"), [UniformRandomRating(seed=1)])


@pytest.mark.unit
def test_backtest_page_renders_all_sections(result):
    page = render_backtest_html(result)
    for heading in (
        "Does it beat the baselines?",
        "Was any edge persistent?",
        "Per-strategy metrics",
        "Rating mix",
    ):
        assert heading in page
    assert page.count("<svg") >= 3
    assert not _EXTERNAL_REF.search(page)


@pytest.mark.unit
def test_verdict_markdown_is_converted_not_printed(result):
    """The wording is shared with the markdown report, so it arrives with ** in it."""
    page = render_backtest_html(result)
    assert "<strong>Verdict</strong>" in page
    assert "**Verdict**" not in page


@pytest.mark.unit
def test_every_chart_is_accompanied_by_its_table(result):
    """Required relief: the rating ramp's neutral step is below 3:1 on the surface."""
    page = render_backtest_html(result)
    assert page.count("<table>") >= 3


@pytest.mark.unit
def test_missing_cutoff_is_called_out(result):
    assert "No knowledge cutoff was supplied" in render_backtest_html(result)


@pytest.mark.unit
def test_interval_chart_marks_significance_without_relying_on_colour():
    """Colour reinforces the verdict; the hover text and table state it in words."""
    svg = interval_chart([("beats", 0.02, 0.01, 0.03, True), ("ties", 0.0, -0.01, 0.01, False)])
    assert "excludes zero, positive" in svg
    assert "interval includes zero" in svg


@pytest.mark.unit
def test_comparison_row_helper_uses_the_baseline_name():
    from tradingagents.webreport.backtest import _comparison_rows

    comparison = PairedComparison(
        strategy="s", baseline="always_buy", n=10, mean_difference=0.01,
        ci_low=0.001, ci_high=0.02, p_value=0.03, confidence=0.95,
    )
    rows = _comparison_rows([comparison])
    assert rows[0][0] == "always_buy"
    assert rows[0][4] is True


@pytest.mark.unit
def test_cumulative_series_is_a_running_mean_not_a_sum():
    """A running sum grows with decision count, so two strategies would not compare."""
    from tradingagents.webreport.backtest import _cumulative_series

    def decision(date, alpha):
        return Decision(
            strategy="s", point=EvalPoint(ticker="A", date=date), rating="Buy", position=1.0,
            outcome=ForwardReturn(
                ticker="A", decision_date=date, entry_date=date, exit_date=date,
                holding_days=5, raw_return=alpha, benchmark_return=0.0,
            ),
        )

    series = _cumulative_series({"s": [decision("2025-01-02", 0.10), decision("2025-02-03", 0.10)]})
    _, points = series[0]
    assert points[0][1] == pytest.approx(0.10)
    assert points[1][1] == pytest.approx(0.10)  # mean stays flat, a sum would double
