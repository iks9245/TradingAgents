"""Refreshing the run index as part of a save.

The index is the only artifact a save writes outside the directory it was
handed, so the two properties under test are where it is allowed to appear and
what happens when writing it goes wrong. As with the HTML, a completed analysis
costs real money: nothing here may turn a successful save into a failed one.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from tradingagents.reporting import write_report_bundle, write_report_tree
from tradingagents.run_index import is_run_directory, write_index

STATE = {
    "market_report": "## Trend\n\nAbove the 50 SMA.\n",
    "investment_debate_state": {"judge_decision": "**Rating**: Buy"},
    "trader_investment_plan": "Half size.",
    "risk_debate_state": {"judge_decision": "**Rating**: Buy"},
}


def _run_dir(reports: Path, ticker: str = "NVDA", stamp: str = "20260814_120000") -> Path:
    return reports / f"{ticker}_{stamp}"


@pytest.mark.unit
def test_saving_a_run_refreshes_the_sibling_index(tmp_path):
    reports = tmp_path / "reports"
    paths = write_report_bundle(STATE, "NVDA", _run_dir(reports))

    assert paths.index == reports / "index.html"
    page = paths.index.read_text(encoding="utf-8")
    assert "NVDA" in page
    assert "<table" in page


@pytest.mark.unit
def test_the_index_lists_earlier_runs_it_did_not_write(tmp_path):
    """A refresh rescans the directory rather than appending to a stale page."""
    reports = tmp_path / "reports"
    write_report_bundle(STATE, "AMD", _run_dir(reports, "AMD", "20260813_090000"))
    paths = write_report_bundle(STATE, "NVDA", _run_dir(reports))

    page = paths.index.read_text(encoding="utf-8")
    assert "AMD" in page
    assert "NVDA" in page


@pytest.mark.unit
def test_the_index_links_the_html_page_written_moments_earlier(tmp_path):
    """Ordering matters: the index picks whichever report form is on disk."""
    reports = tmp_path / "reports"
    paths = write_report_bundle(STATE, "NVDA", _run_dir(reports))

    assert "NVDA_20260814_120000/complete_report.html" in paths.index.read_text(
        encoding="utf-8"
    )


@pytest.mark.unit
def test_a_save_path_of_another_shape_gets_no_index(tmp_path):
    """`--save-to /tmp/run` must not drop an index.html into /tmp."""
    paths = write_report_bundle(STATE, "NVDA", tmp_path / "run")

    assert paths.index is None
    assert not (tmp_path / "index.html").exists()
    assert paths.markdown.exists()


@pytest.mark.unit
def test_index_can_be_disabled_per_call(tmp_path):
    reports = tmp_path / "reports"
    paths = write_report_bundle(STATE, "NVDA", _run_dir(reports), index=False)

    assert paths.index is None
    assert not (reports / "index.html").exists()
    assert paths.markdown.exists()


@pytest.mark.unit
def test_index_can_be_disabled_by_config(tmp_path):
    reports = tmp_path / "reports"
    with patch(
        "tradingagents.dataflows.config.get_config", return_value={"report_index": False}
    ):
        paths = write_report_bundle(STATE, "NVDA", _run_dir(reports))

    assert paths.index is None
    assert not (reports / "index.html").exists()


@pytest.mark.unit
def test_an_index_failure_never_loses_the_report(tmp_path):
    reports = tmp_path / "reports"
    with patch(
        "tradingagents.run_index.write_index", side_effect=OSError("read-only volume")
    ):
        paths = write_report_bundle(STATE, "NVDA", _run_dir(reports))

    assert paths.index is None
    assert paths.markdown.exists()
    assert "Above the 50 SMA" in paths.markdown.read_text(encoding="utf-8")


@pytest.mark.unit
def test_a_damaged_sibling_run_still_reaches_the_index(tmp_path):
    """Scanning tolerates neighbours it cannot parse; the save proceeds."""
    reports = tmp_path / "reports"
    (reports / "INTC_20260101_000000").mkdir(parents=True)  # no report inside

    paths = write_report_bundle(STATE, "NVDA", _run_dir(reports))

    page = paths.index.read_text(encoding="utf-8")
    assert "INTC" in page
    assert "incomplete" in page


@pytest.mark.unit
def test_write_report_tree_refreshes_the_index_too(tmp_path):
    """The legacy entry point returns only markdown but does the same work."""
    reports = tmp_path / "reports"
    markdown = write_report_tree(STATE, "NVDA", _run_dir(reports))

    assert markdown.exists()
    assert (reports / "index.html").exists()


@pytest.mark.unit
@pytest.mark.parametrize(
    "name, expected",
    [
        ("NVDA_20260814_120000", True),
        ("BRK.B_20260814_120000", True),
        ("BTC-USD_20260814_120000", True),
        ("^GSPC_20260814_120000", True),
        ("run", False),
        ("NVDA_20260814", False),
        ("nvda_20260814_120000", False),
        ("index.html", False),
    ],
)
def test_is_run_directory_recognizes_the_scanned_layout(name, expected):
    assert is_run_directory(Path("/anywhere") / name) is expected


@pytest.mark.unit
def test_write_index_creates_a_missing_destination_directory(tmp_path):
    out = write_index(tmp_path / "reports", out=tmp_path / "nested/deeper/index.html")

    assert out.exists()
    assert "<th>Ticker</th>" in out.read_text(encoding="utf-8")
