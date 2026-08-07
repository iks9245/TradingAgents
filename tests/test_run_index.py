"""Run-index coverage built from isolated report trees rather than live output."""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

from tradingagents.agents.utils.structured import UNVALIDATED_MARKER
from tradingagents.run_index import (
    RunRecord,
    main,
    render_index_html,
    render_index_markdown,
    scan_runs,
)


@pytest.mark.unit
def test_import_does_not_load_langchain_stack():
    # Isolation matters because the full suite imports LangChain before this
    # assertion and would make the module count meaningless in this process.
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import tradingagents.run_index, sys; "
                "print(sum(1 for m in sys.modules "
                "if 'langchain' in m or 'langgraph' in m))"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip() == "0"


def _make_run(
    reports_dir: Path,
    run_id: str,
    *,
    report: str | None = None,
    decision: str | None = None,
    trader: str | None = None,
    warnings: str | None = None,
    html: bool = False,
) -> Path:
    run_dir = reports_dir / run_id
    run_dir.mkdir(parents=True)
    if report is not None:
        (run_dir / "complete_report.md").write_text(report, encoding="utf-8")
    if decision is not None:
        (run_dir / "5_portfolio").mkdir()
        (run_dir / "5_portfolio" / "decision.md").write_text(decision, encoding="utf-8")
    if trader is not None:
        (run_dir / "3_trading").mkdir()
        (run_dir / "3_trading" / "trader.md").write_text(trader, encoding="utf-8")
    if warnings is not None:
        (run_dir / "numeric_warnings.md").write_text(warnings, encoding="utf-8")
    if html:
        (run_dir / "complete_report.html").write_text("<html></html>", encoding="utf-8")
    return run_dir


def _report(generated: str = "2026-08-07 23:05:56", code: str | None = None) -> str:
    suffix = "" if code is None else f" · code {code}"
    return f"# Trading Analysis Report: INTC\n\nGenerated: {generated}{suffix}\n\nBody\n"


@pytest.mark.unit
def test_missing_and_empty_reports_directories_return_no_runs(tmp_path):
    assert scan_runs(tmp_path / "missing") == []
    empty = tmp_path / "empty"
    empty.mkdir()
    assert scan_runs(empty) == []


@pytest.mark.unit
def test_run_id_parsing_accepts_supported_symbols_and_skips_junk(tmp_path):
    _make_run(tmp_path, "INTC_20260807_230554", report=_report())
    _make_run(tmp_path, "BRK.B^_20260806_120000", report=_report())
    (tmp_path / "notes").mkdir()

    records = scan_runs(tmp_path)

    assert {record.ticker for record in records} == {"INTC", "BRK.B^"}
    assert all(record.started_at is not None for record in records)


@pytest.mark.unit
def test_header_revision_variants_preserve_unknown_provenance(tmp_path):
    _make_run(tmp_path, "PLAIN_20260807_230554", report=_report(code="3e9a330"))
    _make_run(
        tmp_path,
        "DIRTY_20260807_220000",
        report=_report(code="3e9a330+local-changes"),
    )
    _make_run(
        tmp_path,
        "IMPORTED_20260807_210000",
        report=_report(code="unknown — imported from /path/to/pkg"),
    )
    _make_run(tmp_path, "OLDER_20260807_200000", report=_report())

    records = {record.ticker: record for record in scan_runs(tmp_path)}

    assert records["PLAIN"].generated_at == "2026-08-07 23:05:56"
    assert records["PLAIN"].code_revision == "3e9a330"
    assert records["DIRTY"].code_revision == "3e9a330+local-changes"
    assert "local changes" in records["DIRTY"].statuses
    for ticker in ("IMPORTED", "OLDER"):
        assert records[ticker].code_revision is None
        assert "unknown code" in records[ticker].statuses


@pytest.mark.unit
def test_portfolio_rating_and_target_with_absent_decision_fallback(tmp_path):
    _make_run(
        tmp_path,
        "RATED_20260807_230554",
        report=_report(code="abc1234"),
        decision="**Rating**: Underweight\n\n**Price Target**: 90.0\n",
    )
    _make_run(tmp_path, "ABSENT_20260807_220000", report=_report(code="abc1234"))

    records = {record.ticker: record for record in scan_runs(tmp_path)}

    assert records["RATED"].rating == "Underweight"
    assert records["RATED"].price_target == 90.0
    assert records["ABSENT"].rating is None
    assert records["ABSENT"].price_target is None


@pytest.mark.unit
def test_blockquoted_warning_cannot_poison_real_rating(tmp_path):
    report = (
        _report(code="abc1234")
        + "> **[conflict]** Sell appears in warning text\n\n"
        + "**Rating**: Buy\n"
    )
    _make_run(tmp_path, "SAFE_20260807_230554", report=report)

    assert scan_runs(tmp_path)[0].rating == "Buy"


@pytest.mark.unit
def test_trader_action_reuses_labelled_action_parser(tmp_path):
    _make_run(
        tmp_path,
        "ACTION_20260807_230554",
        report=_report(code="abc1234"),
        trader="> Recommendation: Hold\n\nAction: Sell\n",
    )

    assert scan_runs(tmp_path)[0].action == "Sell"


@pytest.mark.unit
def test_warning_counts_include_breakdown_and_absence(tmp_path):
    _make_run(tmp_path, "NONE_20260807_230554", report=_report(code="abc1234"))
    _make_run(
        tmp_path,
        "ONE_20260807_220000",
        report=_report(code="abc1234"),
        warnings="> **[unit] one**\n",
    )
    _make_run(
        tmp_path,
        "MANY_20260807_210000",
        report=_report(code="abc1234"),
        warnings=(
            "> **[arithmetic] first**\n"
            "> detail\n"
            "> **[conflict] second**\n"
            "> **[arithmetic] third**\n"
        ),
    )

    records = {record.ticker: record for record in scan_runs(tmp_path)}

    assert records["NONE"].warnings == {}
    assert records["NONE"].warning_count == 0
    assert records["ONE"].warnings == {"unit": 1}
    assert records["ONE"].statuses[0] == "1 warning"
    assert records["MANY"].warnings == {"arithmetic": 2, "conflict": 1}
    assert records["MANY"].warning_count == 3


@pytest.mark.unit
def test_unvalidated_sections_count_shared_marker(tmp_path):
    report = _report(code="abc1234") + f"{UNVALIDATED_MARKER}\n{UNVALIDATED_MARKER}\n"
    _make_run(tmp_path, "MARKERS_20260807_230554", report=report)

    assert scan_runs(tmp_path)[0].unvalidated_sections == 2


@pytest.mark.unit
def test_empty_complete_report_is_incomplete_but_still_listed(tmp_path):
    _make_run(tmp_path, "BROKEN_20260807_230554", report="")

    records = scan_runs(tmp_path)

    assert len(records) == 1
    assert records[0].complete is False
    assert records[0].statuses[:2] == ("incomplete", "unknown code")


@pytest.mark.unit
def test_runs_are_ordered_newest_first(tmp_path):
    for run_id in (
        "TWO_20260806_000000",
        "THREE_20260807_000000",
        "ONE_20260805_000000",
    ):
        _make_run(tmp_path, run_id, report=_report(code="abc1234"))

    assert [record.ticker for record in scan_runs(tmp_path)] == ["THREE", "TWO", "ONE"]


def _record(
    tmp_path: Path,
    *,
    ticker: str = "INTC",
    revision: str | None = "abc1234",
    warnings: dict[str, int] | None = None,
    unvalidated: int = 0,
    complete: bool = True,
    rating: str | None = "Buy",
) -> RunRecord:
    run_id = f"{ticker}_20260807_230554"
    path = tmp_path / run_id
    path.mkdir(parents=True)
    return RunRecord(
        run_id=run_id,
        ticker=ticker,
        started_at=datetime(2026, 8, 7, 23, 5, 54),
        path=path,
        generated_at="2026-08-07 23:05:56",
        code_revision=revision,
        rating=rating,
        price_target=90.0,
        action="Buy",
        warnings=warnings or {},
        unvalidated_sections=unvalidated,
        complete=complete,
    )


@pytest.mark.unit
def test_markdown_mixed_records_show_every_status_label(tmp_path):
    records = [
        _record(tmp_path, ticker="OK"),
        _record(
            tmp_path,
            ticker="ISSUES",
            revision=None,
            warnings={"unit": 2},
            unvalidated=3,
            complete=False,
        ),
        _record(tmp_path, ticker="DIRTY", revision="abc1234+local-changes"),
        _record(tmp_path, ticker="SINGLE", warnings={"conflict": 1}),
    ]

    markdown = render_index_markdown(records)

    for label in (
        "ok",
        "incomplete",
        "2 warnings",
        "3 unvalidated",
        "unknown code",
        "local changes",
        "1 warning",
    ):
        assert label in markdown


@pytest.mark.unit
def test_html_is_self_contained_escaped_and_prefers_html_report(tmp_path):
    record = _record(tmp_path, ticker="A<&", revision="path<&", rating="Sell")
    (record.path / "complete_report.html").write_text("page", encoding="utf-8")

    page = render_index_html([record])

    assert "http://" not in page
    assert "https://" not in page
    assert "A&lt;&amp;" in page
    assert "path&lt;&amp;" in page
    escaped_run_id = "A&lt;&amp;_20260807_230554"
    assert f'{escaped_run_id}/complete_report.html' in page
    assert "var(--rating-sell)" in page


@pytest.mark.unit
def test_main_writes_output_and_missing_directory_still_succeeds(tmp_path):
    reports = tmp_path / "reports"
    _make_run(reports, "INTC_20260807_230554", report=_report(code="abc1234"))
    html_out = tmp_path / "runs.html"

    assert main([str(reports), "-o", str(html_out)]) == 0
    assert "INTC" in html_out.read_text(encoding="utf-8")

    missing = tmp_path / "missing"
    markdown_out = tmp_path / "empty.md"
    assert main([str(missing), "--markdown", "-o", str(markdown_out)]) == 0
    text = markdown_out.read_text(encoding="utf-8")
    assert "| Ticker | Started | Rating | Target | Code | Status |" in text
    assert len(text.splitlines()) == 2
