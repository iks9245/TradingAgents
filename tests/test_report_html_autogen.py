"""Automatic HTML generation after a saved run.

The property that matters most here is the failure mode: a completed analysis
costs real money, so a rendering bug must never turn a successful save into a
lost report.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from tradingagents.reporting import ReportPaths, write_report_bundle, write_report_tree

STATE = {
    "market_report": "## Trend\n\nAbove the 50 SMA.\n",
    "sentiment_report": "Bullish chatter.\n",
    "investment_debate_state": {
        "bull_history": "Bull Analyst: up",
        "bear_history": "Bear Analyst: down",
        "judge_decision": "**Rating**: Hold",
    },
    "trader_investment_plan": "Half size.",
    "risk_debate_state": {"judge_decision": "**Rating**: Hold"},
}


@pytest.mark.unit
def test_html_is_written_beside_the_markdown_by_default(tmp_path):
    paths = write_report_bundle(STATE, "NVDA", tmp_path / "run")

    assert paths.markdown.name == "complete_report.md"
    assert paths.html is not None
    assert paths.html.name == "complete_report.html"
    assert paths.html.parent == paths.markdown.parent
    assert paths.html.exists()


@pytest.mark.unit
def test_generated_html_contains_the_report_content(tmp_path):
    paths = write_report_bundle(STATE, "NVDA", tmp_path / "run")
    page = paths.html.read_text(encoding="utf-8")

    assert "NVDA" in page
    assert "Above the 50 SMA" in page
    assert "<style>" in page  # self-contained, not linking a stylesheet


@pytest.mark.unit
def test_html_can_be_disabled_per_call(tmp_path):
    paths = write_report_bundle(STATE, "NVDA", tmp_path / "run", html=False)

    assert paths.html is None
    assert paths.markdown.exists()
    assert not (tmp_path / "run" / "complete_report.html").exists()


@pytest.mark.unit
def test_html_can_be_disabled_by_config(tmp_path):
    """``report_html: false`` in config, or TRADINGAGENTS_REPORT_HTML=false."""
    from tradingagents.dataflows.config import set_config

    set_config({"report_html": False})
    assert write_report_bundle(STATE, "NVDA", tmp_path / "run").html is None


@pytest.mark.unit
def test_explicit_argument_beats_config(tmp_path):
    from tradingagents.dataflows.config import set_config

    set_config({"report_html": False})
    assert write_report_bundle(STATE, "NVDA", tmp_path / "run", html=True).html is not None


@pytest.mark.unit
def test_a_renderer_failure_never_loses_the_markdown(tmp_path):
    """The whole point of the try/except: a completed run cost real money."""
    with patch(
        "tradingagents.webreport.markdown_to_page", side_effect=RuntimeError("boom")
    ):
        paths = write_report_bundle(STATE, "NVDA", tmp_path / "run")

    assert paths.html is None                     # reported honestly, not faked
    assert paths.markdown.exists()                # and the markdown survived
    assert "Above the 50 SMA" in paths.markdown.read_text(encoding="utf-8")


@pytest.mark.unit
def test_renderer_failure_is_logged(tmp_path, caplog):
    with patch(
        "tradingagents.webreport.markdown_to_page", side_effect=RuntimeError("boom")
    ):
        write_report_bundle(STATE, "NVDA", tmp_path / "run")

    assert any("HTML report" in record.message for record in caplog.records)


@pytest.mark.unit
def test_write_report_tree_keeps_its_original_contract(tmp_path):
    """Existing callers get a Path back, exactly as before."""
    result = write_report_tree(STATE, "NVDA", tmp_path / "run")

    assert isinstance(result, Path)
    assert result.name == "complete_report.md"
    # ...and now also produce the HTML as a side effect.
    assert result.with_suffix(".html").exists()


@pytest.mark.unit
def test_section_files_are_still_written(tmp_path):
    """Adding HTML must not disturb the existing tree layout."""
    write_report_bundle(STATE, "NVDA", tmp_path / "run")
    assert (tmp_path / "run" / "1_analysts" / "market.md").exists()
    assert (tmp_path / "run" / "1_analysts" / "sentiment.md").exists()


@pytest.mark.unit
def test_report_paths_defaults_html_to_none():
    assert ReportPaths(markdown=Path("x.md")).html is None


@pytest.mark.unit
def test_graph_save_reports_produces_html(tmp_path):
    """The programmatic path gets it too, via the same shared writer."""
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    state = dict(STATE, company_of_interest="NVDA")
    path = TradingAgentsGraph.save_reports(
        object.__new__(TradingAgentsGraph), state, "NVDA", save_path=tmp_path / "run"
    )
    assert path.with_suffix(".html").exists()


# --- CLI wiring ---------------------------------------------------------


@pytest.mark.unit
def test_cli_config_honours_the_html_flag():
    from cli.main import _build_run_config

    selections = {
        "research_depth": 1, "shallow_thinker": "m", "deep_thinker": "m",
        "backend_url": "u", "llm_provider": "OpenAI",
    }
    assert _build_run_config(selections, None, False)["report_html"] is False
    assert _build_run_config(selections, None, True)["report_html"] is True


@pytest.mark.unit
def test_cli_config_leaves_html_alone_when_flag_omitted():
    """Omitting --html/--no-html must preserve the env var / default."""
    from cli.main import _build_run_config

    selections = {
        "research_depth": 1, "shallow_thinker": "m", "deep_thinker": "m",
        "backend_url": "u", "llm_provider": "OpenAI",
    }
    config = _build_run_config(selections, None, None)
    assert config["report_html"] is True  # the DEFAULT_CONFIG value, not overridden


@pytest.mark.unit
def test_cli_exposes_the_html_flag():
    from typer.main import get_command

    from cli.main import app

    command = get_command(app)
    # A single-command Typer app compiles to a Command, not a Group.
    command = command.commands["analyze"] if hasattr(command, "commands") else command
    assert "html" in {p.name for p in command.params}


@pytest.mark.unit
def test_env_var_controls_report_html(monkeypatch):
    import importlib

    import tradingagents.default_config as default_config

    monkeypatch.setenv("TRADINGAGENTS_REPORT_HTML", "false")
    reloaded = importlib.reload(default_config)
    try:
        assert reloaded.DEFAULT_CONFIG["report_html"] is False
    finally:
        monkeypatch.delenv("TRADINGAGENTS_REPORT_HTML", raising=False)
        importlib.reload(default_config)


@pytest.mark.unit
def test_html_does_not_repeat_the_title_and_timestamp(tmp_path):
    """The page shell renders both; leaving them in the body prints them twice."""
    paths = write_report_bundle(STATE, "NVDA", tmp_path / "run")
    page = paths.html.read_text(encoding="utf-8")

    # Exactly two: the <title> tag and the page <h1>. The markdown body must
    # not contribute a third, nor a second <h1>.
    assert page.count("Trading Analysis Report: NVDA") == 2
    assert page.count("<h1>") == 1
    assert page.count("Generated") == 1
    # The markdown itself keeps its header — only the HTML body drops it.
    assert paths.markdown.read_text(encoding="utf-8").startswith("# Trading Analysis Report:")


@pytest.mark.unit
def test_header_stripping_leaves_foreign_markdown_alone():
    """Only this module's own header is peeled, never a report's first section."""
    from tradingagents.reporting import _split_own_header

    body, generated = _split_own_header("## Market Analyst\n\ncontent\n")
    assert body == "## Market Analyst\n\ncontent\n"
    assert generated == ""

    body, generated = _split_own_header("# Trading Analysis Report: X\n\nGenerated: now\n\n## S\n")
    assert body == "## S\n"
    assert generated == "Generated: now"
