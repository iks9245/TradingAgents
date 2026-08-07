"""The analyze command can run unattended when its choices come from flags/env."""

from __future__ import annotations

import datetime
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

import cli.main as m
from tradingagents.reporting import ReportPaths
from tradingagents.run_index import scan_runs


class _QuietLive:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


class _FakeTradingAgentsGraph:
    instances = []

    def __init__(self, selected_analysts, config, **kwargs):
        self.selected_analysts = selected_analysts
        self.config = config
        self.propagator = self
        self.graph = self
        self.ticker = None
        self.analysis_date = None
        self.instances.append(self)

    def resolve_instrument_context(self, ticker, asset_type):
        self.ticker = ticker
        return {"symbol": ticker, "asset_type": asset_type}

    def resolve_verified_evidence(self, ticker, trade_date, asset_type="stock"):
        return (f"market block for {ticker}", f"fundamentals block for {ticker}")

    def create_initial_state(
        self,
        ticker,
        analysis_date,
        *,
        asset_type,
        instrument_context,
        verified_market_block=None,
        verified_fundamentals_block=None,
    ):
        self.ticker = ticker
        self.analysis_date = analysis_date
        # Recorded rather than dropped: the blocks are what keeps the debate
        # able to check a figure, so a scripted run losing them should fail
        # here rather than in a report nobody reads.
        self.verified_blocks = (verified_market_block, verified_fundamentals_block)
        return {}

    def get_graph_args(self, **kwargs):
        return {}

    def stream(self, state, **kwargs):
        yield {
            "market_report": "Market report",
            "sentiment_report": "Sentiment report",
            "news_report": "News report",
            "fundamentals_report": "Fundamentals report",
            "investment_debate_state": {"judge_decision": "Research decision"},
            "trader_investment_plan": "Trading plan",
            "risk_debate_state": {"judge_decision": "Portfolio decision"},
            "final_trade_decision": "Portfolio decision",
        }


@pytest.fixture
def non_interactive_cli(monkeypatch, tmp_path):
    original_buffer_methods = {
        name: getattr(m.message_buffer, name)
        for name in ("add_message", "add_tool_call", "update_report_section")
    }
    _FakeTradingAgentsGraph.instances.clear()

    env = {
        "TRADINGAGENTS_LLM_PROVIDER": "ollama",
        "TRADINGAGENTS_QUICK_THINK_LLM": "local-quick",
        "TRADINGAGENTS_DEEP_THINK_LLM": "local-deep",
        "TRADINGAGENTS_OUTPUT_LANGUAGE": "English",
        "TRADINGAGENTS_MAX_DEBATE_ROUNDS": "1",
        "TRADINGAGENTS_MAX_RISK_ROUNDS": "1",
    }
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    fake_config = dict(m.DEFAULT_CONFIG)
    fake_config.update(
        {
            "llm_provider": "ollama",
            "backend_url": "http://localhost:11434/v1",
            "quick_think_llm": "local-quick",
            "deep_think_llm": "local-deep",
            "output_language": "English",
            "max_debate_rounds": 1,
            "max_risk_discuss_rounds": 1,
            "results_dir": str(tmp_path / "internal-results"),
            "report_html": False,
        }
    )
    monkeypatch.setattr(m, "DEFAULT_CONFIG", fake_config)
    monkeypatch.setattr(m, "TradingAgentsGraph", _FakeTradingAgentsGraph)
    monkeypatch.setattr(m, "Live", _QuietLive)
    monkeypatch.setattr(m, "update_display", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        m,
        "fetch_announcements",
        lambda: pytest.fail("announcements must not be fetched"),
    )

    saved_paths = []

    def fake_save(final_state, ticker, save_path):
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        markdown = save_path / "complete_report.md"
        markdown.write_text(f"# Trading Analysis Report: {ticker}\n", encoding="utf-8")
        saved_paths.append(save_path)
        return ReportPaths(markdown=markdown)

    monkeypatch.setattr(m, "save_report_to_disk", fake_save)
    monkeypatch.chdir(tmp_path)

    yield SimpleNamespace(
        runner=CliRunner(),
        saved_paths=saved_paths,
        graph_instances=_FakeTradingAgentsGraph.instances,
        config=fake_config,
    )

    for name, method in original_buffer_methods.items():
        setattr(m.message_buffer, name, method)


def _raise_on_prompt(*args, **kwargs):
    raise AssertionError("interactive prompt called")


def test_non_interactive_run_never_prompts(monkeypatch, non_interactive_cli):
    import questionary

    monkeypatch.setattr(m.typer, "prompt", _raise_on_prompt)
    for name in ("text", "password", "select", "checkbox"):
        monkeypatch.setattr(questionary, name, _raise_on_prompt)

    result = non_interactive_cli.runner.invoke(
        m.app, ["--non-interactive", "--ticker", "INTC"]
    )

    assert result.exit_code == 0, result.output
    assert non_interactive_cli.saved_paths


def test_non_interactive_never_prompts_without_any_env_vars(
    monkeypatch, non_interactive_cli
):
    """--non-interactive must suffice on its own, with nothing exported.

    The fixture sets all six TRADINGAGENTS_* vars, so every other test here
    travels the pre-existing env-skip path and cannot see whether the flag alone
    works. It did not: output language, research depth, provider and thinking
    agents each still reached for a prompt, and against a closed stdin the run
    died with "Input is not a terminal" — naming neither the step responsible
    nor the variable that would have settled it.
    """
    import questionary

    for name in (
        "TRADINGAGENTS_LLM_PROVIDER",
        "TRADINGAGENTS_QUICK_THINK_LLM",
        "TRADINGAGENTS_DEEP_THINK_LLM",
        "TRADINGAGENTS_OUTPUT_LANGUAGE",
        "TRADINGAGENTS_MAX_DEBATE_ROUNDS",
        "TRADINGAGENTS_MAX_RISK_ROUNDS",
    ):
        monkeypatch.delenv(name, raising=False)

    monkeypatch.setattr(m.typer, "prompt", _raise_on_prompt)
    for name in ("text", "password", "select", "checkbox"):
        monkeypatch.setattr(questionary, name, _raise_on_prompt)

    result = non_interactive_cli.runner.invoke(
        m.app, ["--non-interactive", "--ticker", "INTC"]
    )

    assert result.exit_code == 0, result.output
    assert non_interactive_cli.saved_paths


def test_non_interactive_requires_ticker(non_interactive_cli):
    result = non_interactive_cli.runner.invoke(m.app, ["--non-interactive"])

    assert result.exit_code == 2
    assert "--ticker" in result.output


@pytest.mark.parametrize("bad_date", ["2099-01-01", "08/08/2026", "2026-8-8"])
def test_non_interactive_rejects_bad_dates(non_interactive_cli, bad_date):
    result = non_interactive_cli.runner.invoke(
        m.app,
        ["--non-interactive", "--ticker", "INTC", "--date", bad_date],
    )

    assert result.exit_code == 2
    assert "date" in result.output.lower()


def test_omitted_date_defaults_to_today(non_interactive_cli):
    result = non_interactive_cli.runner.invoke(
        m.app, ["--non-interactive", "--ticker", "INTC", "--no-save"]
    )

    assert result.exit_code == 0, result.output
    assert non_interactive_cli.graph_instances[-1].analysis_date == datetime.date.today().isoformat()


def test_ticker_is_normalized(non_interactive_cli):
    result = non_interactive_cli.runner.invoke(
        m.app, ["--non-interactive", "--ticker", "btcusd", "--no-save"]
    )

    assert result.exit_code == 0, result.output
    assert non_interactive_cli.graph_instances[-1].ticker == "BTC-USD"


def test_explicit_analysts_are_selected_exactly(non_interactive_cli):
    result = non_interactive_cli.runner.invoke(
        m.app,
        [
            "--non-interactive",
            "--ticker",
            "INTC",
            "--analysts",
            "market,news",
            "--no-save",
        ],
    )

    assert result.exit_code == 0, result.output
    assert non_interactive_cli.graph_instances[-1].selected_analysts == [
        "market",
        "news",
    ]


def test_unknown_analyst_lists_valid_names(non_interactive_cli):
    result = non_interactive_cli.runner.invoke(
        m.app,
        [
            "--non-interactive",
            "--ticker",
            "INTC",
            "--analysts",
            "bogus",
        ],
    )

    assert result.exit_code == 2
    for name in ("market", "social", "news", "fundamentals"):
        assert name in result.output


def test_omitted_analysts_selects_all_valid_for_asset(non_interactive_cli):
    result = non_interactive_cli.runner.invoke(
        m.app, ["--non-interactive", "--ticker", "BTC-USD", "--no-save"]
    )

    assert result.exit_code == 0, result.output
    assert non_interactive_cli.graph_instances[-1].selected_analysts == [
        "market",
        "social",
        "news",
    ]


def test_save_to_and_no_save(non_interactive_cli, tmp_path):
    destination = tmp_path / "chosen-report"
    result = non_interactive_cli.runner.invoke(
        m.app,
        [
            "--non-interactive",
            "--ticker",
            "INTC",
            "--save-to",
            str(destination),
        ],
    )

    assert result.exit_code == 0, result.output
    assert (destination / "complete_report.md").is_file()

    non_interactive_cli.saved_paths.clear()
    result = non_interactive_cli.runner.invoke(
        m.app, ["--non-interactive", "--ticker", "INTC", "--no-save"]
    )
    assert result.exit_code == 0, result.output
    assert non_interactive_cli.saved_paths == []


def test_a_scripted_run_still_hands_the_debate_its_verified_evidence(
    non_interactive_cli,
):
    """The evidence blocks reach the graph on this path too, not just the prompts."""
    result = non_interactive_cli.runner.invoke(
        m.app, ["--non-interactive", "--ticker", "INTC"]
    )

    assert result.exit_code == 0, result.output
    graph = non_interactive_cli.graph_instances[-1]
    assert graph.verified_blocks == (
        "market block for INTC",
        "fundamentals block for INTC",
    )


def test_default_save_path_is_indexable(non_interactive_cli, tmp_path):
    result = non_interactive_cli.runner.invoke(
        m.app, ["--non-interactive", "--ticker", "INTC"]
    )

    assert result.exit_code == 0, result.output
    save_path = non_interactive_cli.saved_paths[-1]
    assert re.fullmatch(r"[A-Z0-9.\-^]+_\d{8}_\d{6}", save_path.name)
    assert any(record.path == save_path for record in scan_runs(tmp_path / "reports"))
    assert result.output.rstrip().endswith(str(save_path.resolve()))


def test_failed_save_exits_nonzero(monkeypatch, non_interactive_cli):
    """A scripted run must not report success when the report was never written.

    The caller's next move is to open the save path; exiting 0 sends it looking
    for a directory that does not exist.
    """
    def failing_save(final_state, ticker, save_path):
        raise OSError("disk full")

    monkeypatch.setattr(m, "save_report_to_disk", failing_save)
    result = non_interactive_cli.runner.invoke(
        m.app, ["--non-interactive", "--ticker", "INTC"]
    )

    assert result.exit_code == 1, result.output


def test_missing_api_key_exits_without_prompt(monkeypatch, non_interactive_cli):
    monkeypatch.setenv("TRADINGAGENTS_LLM_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    non_interactive_cli.config.update(
        {"llm_provider": "openai", "backend_url": "https://api.openai.com/v1"}
    )

    import cli.utils as utils

    monkeypatch.setattr(utils.questionary, "password", _raise_on_prompt)
    result = non_interactive_cli.runner.invoke(
        m.app, ["--non-interactive", "--ticker", "INTC", "--no-save"]
    )

    assert result.exit_code == 2
    assert "OPENAI_API_KEY" in result.output


def test_analysis_failure_is_one_line_without_traceback(
    monkeypatch, non_interactive_cli
):
    def fail_stream(self, state, **kwargs):
        raise RuntimeError("provider failed\nrequest rejected")
        yield  # pragma: no cover - makes this a generator like graph.stream

    monkeypatch.setattr(_FakeTradingAgentsGraph, "stream", fail_stream)
    result = non_interactive_cli.runner.invoke(
        m.app, ["--non-interactive", "--ticker", "INTC", "--no-save"]
    )

    assert result.exit_code == 1
    assert "Error: analysis failed: provider failed request rejected" in result.output
    assert "Traceback" not in result.output


def test_prefilled_ticker_keeps_remaining_steps_interactive(
    monkeypatch, non_interactive_cli
):
    prompted = False

    def date_prompt():
        nonlocal prompted
        prompted = True
        raise RuntimeError("date prompt reached")

    monkeypatch.setattr(m, "fetch_announcements", lambda: None)
    monkeypatch.setattr(m, "display_announcements", lambda *args: None)
    monkeypatch.setattr(m, "get_ticker", _raise_on_prompt)
    monkeypatch.setattr(m, "get_analysis_date", date_prompt)

    result = non_interactive_cli.runner.invoke(
        m.app, ["--ticker", "INTC", "--no-save", "--no-display"]
    )

    assert prompted
    assert isinstance(result.exception, RuntimeError)
