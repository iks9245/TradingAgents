"""Unit coverage for deterministic numeric linting of assembled reports."""

import pytest

from tradingagents.report_lint import lint_report, render_warning_block
from tradingagents.reporting import write_report_tree


@pytest.mark.unit
def test_operating_margin_defect_is_recomputed():
    findings = lint_report("營業利潤率 14.4%  (36.94 / 346.39)")
    assert any(finding.kind == "arithmetic" and "10.66%" in finding.summary for finding in findings)


@pytest.mark.unit
def test_net_margin_defect_is_recomputed():
    findings = lint_report("淨利潤率 13.4%  (43.34 / 346.39)")
    assert any(finding.kind == "arithmetic" and "12.51%" in finding.summary for finding in findings)


@pytest.mark.unit
def test_correct_division_and_rounding_tolerance_are_accepted():
    assert lint_report("10.66%  (3,694 / 34,639)") == []
    assert lint_report("49.52%  (17,152 / 34,639)") == []


@pytest.mark.unit
def test_percent_multiple_form_is_recomputed():
    assert lint_report("6.01%  (= 0.0601x)") == []
    findings = lint_report("6.01%  (= 6.0100x)")
    assert any(finding.kind == "arithmetic" for finding in findings)


@pytest.mark.unit
def test_ratio_unit_confusion_is_reported():
    findings = lint_report("債務權益比 6.01%\n債務權益比 6.01 倍")
    assert any(finding.kind == "unit" for finding in findings)


@pytest.mark.unit
def test_conflicting_moving_averages_are_reported_but_nearby_values_are_not():
    findings = lint_report("50日SMA 514.33\n50 日均價 512.95")
    conflict = next(finding for finding in findings if finding.kind == "conflict")
    assert "514.33" in conflict.detail and "512.95" in conflict.detail
    assert not lint_report("50日SMA 514.33\n50 日均價 約 514")


@pytest.mark.unit
@pytest.mark.parametrize(
    "text",
    [
        # "1.5x ATR" is a multiplier in prose, not a second reading of the ATR.
        # Treating it as one would flag correct risk-management wording.
        "ATR 40.39\n止損建議為 1.5 倍 ATR",
        "ATR 40.39\nstop at 1.5x ATR",
    ],
)
def test_multiplier_prose_is_not_a_second_reading(text):
    assert lint_report(text) == []


@pytest.mark.unit
def test_genuinely_different_atr_values_still_conflict():
    findings = lint_report("ATR 40.39\nATR 36.10")
    assert any(finding.kind == "conflict" for finding in findings)


@pytest.mark.unit
def test_moving_average_tolerance_is_tight_enough_for_the_shipped_defect():
    # The shipped 514.33 / 512.95 spread is 0.27%. A 1% bound — fine for ratios
    # quoted in prose — would miss it entirely, so price levels get 0.15%.
    assert lint_report("50日SMA 514.33\n50 日均價 512.95")
    # ...while a plainly rounded restatement stays quiet.
    assert lint_report("50日SMA 514.33\n50 日均價 514.30") == []


@pytest.mark.unit
class TestValueExtractionPrecision:
    """Guards against phantom conflicts, which cost the block its credibility.

    Every case here was produced by running the linter over the real AMD report
    before these filters existed.
    """

    def test_value_must_be_on_the_alias_line(self):
        # Narrative mention followed by an ordered-list marker on the next line.
        # Reading across the newline made "2." a 50-day average reading.
        assert lint_report("價格在50日均线附近整理。\n2. 高波動性可能導致假突破\n| 50日SMA | 514.33 |") == []

    def test_a_number_that_names_another_metric_is_not_a_value(self):
        # "但远高于200日均价" after a 50-day alias: the 200 is the next metric's
        # label, not this metric's value.
        assert lint_report("| 50日均价 | 514.33 |\n當前價格接近50日均价，但远高于200日均价。") == []

    def test_combined_table_cell_is_skipped_rather_than_guessed(self):
        # "50日/200日均價 | 512.95/313.16" packs two metrics into one cell; which
        # number is which is not recoverable positionally.
        assert lint_report("| 50日/200日均價 | 512.95/313.16美元 |") == []

    def test_one_hedged_mention_does_not_silence_a_real_conflict(self):
        # The 512.95 / 514.33 conflict must survive an "約 514" elsewhere in the
        # report — a hedge sits out the comparison, it does not excuse it.
        findings = lint_report(
            "| 50日SMA | 514.33 |\n關注50日均线（约514美元）\n| 50日均价 | 512.95美元 |"
        )
        assert any(finding.kind == "conflict" for finding in findings)


@pytest.mark.unit
class TestValueExtractionPrecisionII:
    """Second round of extraction guards, from the 2026-08-06 INTC report.

    Every warning that report produced was a false positive, and all three had
    distinct causes.
    """

    def test_a_number_followed_by_a_comma_is_not_truncated(self):
        from tradingagents.report_lint import _NUMBER_RE

        # "$111.52," used to match "111", which then "conflicted" with 111.52.
        assert _NUMBER_RE.search("at $111.52, confirming").group(1) == "111.52"
        assert _NUMBER_RE.search("ATR of 8.32, and").group(1) == "8.32"
        # ...while genuine thousands separators still parse whole.
        assert _NUMBER_RE.search("1,234.56 total").group(1) == "1,234.56"
        assert _NUMBER_RE.search("1,234 units").group(1) == "1,234"

    def test_comma_truncation_no_longer_invents_a_conflict(self):
        assert lint_report("Price sits below its 50-day SMA at $111.52, confirming weakness.") == []

    def test_number_in_a_separate_clause_is_not_a_reading(self):
        # "$100B capital programme" is capex prose, not a leverage reading.
        assert lint_report(
            "| Debt to Equity | 49.00% |\n"
            "Intel has a high debt-to-equity ratio, undertaking a **$100B+ capital programme**."
        ) == []

    def test_price_metric_quoted_as_a_percentage_is_not_a_reading(self):
        # "ATR of 8.2%" is ATR as a share of price, not ATR in dollars.
        assert lint_report("ATR is 8.32 today.\nWith an ATR of 8.2%, use protective puts.") == []

    def test_parenthesised_qualifier_still_binds_the_value(self):
        # "(MRQ):" must not break the label-to-value binding.
        findings = lint_report(
            "- **Debt to Equity (MRQ):** 49.00%\n- Debt to Equity (MRQ): 61.00%"
        )
        assert any(f.kind == "conflict" for f in findings)

    def test_relinting_a_saved_report_does_not_read_its_own_warnings(self):
        # The block's evidence line lists every value it found; re-linting fed
        # those back in as fresh readings and manufactured a conflict.
        body = "ATR is 8.32 today."
        saved = (
            "# Trading Analysis Report: INTC\n\n"
            "> ## ⚠️ Numeric consistency warnings\n>\n"
            "> **[conflict] atr is stated as both 8 and 8.32**\n"
            "> Distinct values found for atr: 8, 8.2, 8.32.\n\n" + body
        )
        assert lint_report(saved) == []


@pytest.mark.unit
def test_empty_and_non_numeric_reports_are_clean():
    assert lint_report("") == []
    assert lint_report("A prose report with no figures.") == []


@pytest.mark.unit
def test_warning_block_is_empty_for_a_clean_report():
    assert render_warning_block([]) == ""


@pytest.mark.unit
def test_writer_inserts_and_persists_warnings_only_when_needed(tmp_path):
    clean = write_report_tree({"market_report": "10.66%  (3,694 / 34,639)"}, "AAPL", tmp_path / "clean")
    assert "Numeric consistency warnings" not in clean.read_text()

    warned_dir = tmp_path / "warned"
    warned = write_report_tree({"market_report": "14.4%  (36.94 / 346.39)"}, "AAPL", warned_dir)
    assert "Numeric consistency warnings" in warned.read_text()
    assert (warned_dir / "numeric_warnings.md").exists()


@pytest.mark.unit
def test_malformed_input_never_raises():
    malformed = "(" + "1,2,3 / / 4" + "9" * 50_000
    assert isinstance(lint_report(malformed), list)
