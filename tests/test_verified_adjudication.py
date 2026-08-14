"""Checking a report's figures against the run's own verified market snapshot.

The spread check can only say that a report disagrees with itself, which leaves
two gaps. It cannot say which value is right, and it cannot see a figure that is
wrong but stated consistently — there is no spread to detect. Both are closed by
letting the last gate read the same block the debate reads.
"""

from pathlib import Path

import pytest

from tradingagents.agents.utils.verified_evidence import verified_figures_from_state
from tradingagents.dataflows.market_data_validator import (
    _LEVEL_LINES,
    TradeReference,
    parse_trade_reference_block,
    render_trade_reference_block,
)
from tradingagents.report_lint import lint_report, render_warning_block
from tradingagents.reporting import write_report_bundle

VERIFIED = {"sma50": 110.60, "sma200": 68.57, "atr": 8.09}


def _reference(**overrides) -> TradeReference:
    fields = {
        "symbol": "INTC",
        "as_of": "2026-08-07",
        "bar_status": "final",
        "close": 99.43,
        "atr": 8.09,
        "ema10": 101.2,
        "sma50": 110.60,
        "sma200": 68.57,
        "week52_high": 1142.35,
        "week52_low": 19.61,
    }
    fields.update(overrides)
    return TradeReference(**fields)


# --- reading the block back ------------------------------------------------


@pytest.mark.unit
def test_every_rendered_level_parses_back_to_its_own_value():
    """Render and parse share one table, so a new level cannot escape the parser."""
    reference = _reference()
    figures = parse_trade_reference_block(
        render_trade_reference_block(reference, include_proposal_rule=False)
    )

    assert figures == {field: getattr(reference, field) for _, field in _LEVEL_LINES}


@pytest.mark.unit
def test_a_thousands_separator_survives_the_round_trip():
    block = render_trade_reference_block(_reference(), include_proposal_rule=False)

    assert "- 52-week high: 1,142.35" in block
    assert parse_trade_reference_block(block)["week52_high"] == 1142.35


@pytest.mark.unit
def test_an_unavailable_level_is_absent_rather_than_zero():
    """"The source says nothing" and "the source says 0" must stay distinguishable."""
    figures = parse_trade_reference_block(
        render_trade_reference_block(_reference(ema10=None), include_proposal_rule=False)
    )

    assert "ema10" not in figures
    assert figures["sma50"] == 110.60


@pytest.mark.unit
@pytest.mark.parametrize("block", ["", "   ", "not a block at all", None, 42])
def test_an_unusable_block_yields_no_figures(block):
    assert parse_trade_reference_block(block) == {}


@pytest.mark.unit
def test_the_unavailable_notice_yields_no_figures():
    assert parse_trade_reference_block(
        render_trade_reference_block(None, include_proposal_rule=False)
    ) == {}


# --- reading them off the state --------------------------------------------


@pytest.mark.unit
def test_figures_come_off_the_state_the_debate_already_reads():
    state = {
        "verified_market_block": render_trade_reference_block(
            _reference(), include_proposal_rule=False
        )
    }

    assert verified_figures_from_state(state)["sma50"] == 110.60


@pytest.mark.unit
@pytest.mark.parametrize("state", [{}, {"verified_market_block": ""}, {"verified_market_block": None}])
def test_a_state_without_a_block_adjudicates_nothing(state):
    assert verified_figures_from_state(state) == {}


# --- adjudication ----------------------------------------------------------


@pytest.mark.unit
def test_one_wrong_figure_stated_consistently_is_caught():
    """The case the spread check structurally cannot see: nothing to disagree with."""
    report = (
        "The 50-day SMA sits at 111.05.\n\n"
        "Reclaiming the 50-day moving average of 111.05 would flip the trend.\n\n"
        "Entry on a close above the 50-day SMA (111.05).\n"
    )

    assert lint_report(report) == []
    findings = lint_report(report, verified=VERIFIED)
    assert [finding.kind for finding in findings] == ["contradiction"]
    assert "stated as 111.05" in findings[0].summary
    assert "verified value is 110.6" in findings[0].summary


@pytest.mark.unit
def test_a_faithful_rounding_is_not_a_contradiction():
    """"ATR is 8" against a verified 8.09 is exact at the precision offered."""
    assert lint_report("ATR is 8 for this instrument.", verified=VERIFIED) == []


@pytest.mark.unit
def test_a_last_cent_difference_is_not_a_contradiction():
    """110.59 against a verified 110.60 is the same figure, not a second one."""
    assert lint_report("The 50-day SMA is 110.59.", verified=VERIFIED) == []


@pytest.mark.unit
def test_a_rounding_beside_its_precise_twin_produces_no_conflict():
    """Adjudication replaces the spread check rather than running alongside it.

    Both statements read the source correctly, so a report saying "ATR 8.09" in
    one section and "ATR is 8" in another is not inconsistent — but the spread
    between them exceeds the default tolerance, so the older check called it one.
    """
    report = "ATR (daily volatility): 8.09.\n\nStops sit roughly 1 ATR, about 8, below.\n"

    assert lint_report(report, verified=VERIFIED) == []


@pytest.mark.unit
def test_the_wrong_value_is_named_among_correct_ones():
    report = (
        "The 50-day SMA is 110.60.\n\n"
        "Price closed below the 50-day SMA at 99.43.\n\n"
        "The 50-day moving average remains 110.60.\n"
    )

    findings = lint_report(report, verified=VERIFIED)

    assert len(findings) == 1
    assert "stated as 99.43" in findings[0].summary
    assert "99.43" in findings[0].detail
    assert "110.60" not in findings[0].detail.partition("Contradicting")[2]


@pytest.mark.unit
def test_the_furthest_statement_leads_when_several_disagree():
    report = "50-day SMA of 111.05.\n\nElsewhere the 50-day SMA is 130.00.\n"

    findings = lint_report(report, verified=VERIFIED)

    assert "stated as 130.00" in findings[0].summary
    assert "111.05" in findings[0].detail


# --- falling back ----------------------------------------------------------


@pytest.mark.unit
def test_without_a_snapshot_the_spread_check_is_unchanged():
    report = "The 50-day SMA is 514.33.\n\nThe 50-day moving average is 512.95.\n"

    findings = lint_report(report)

    assert [finding.kind for finding in findings] == ["conflict"]
    assert "both 512.95 and 514.33" in findings[0].summary


@pytest.mark.unit
def test_a_metric_the_snapshot_does_not_cover_still_uses_the_spread_check():
    """Only the market block contributes, so leverage ratios keep the old check."""
    report = "Debt-to-equity is 49.00.\n\nDebt-to-equity stands at 100.\n"

    findings = lint_report(report, verified=VERIFIED)

    assert any(finding.kind == "conflict" for finding in findings)


@pytest.mark.unit
@pytest.mark.parametrize("verified", ["110.60", 110.60, ["sma50"], None])
def test_an_unusable_verified_argument_degrades_to_the_spread_check(verified):
    report = "The 50-day SMA is 514.33.\n\nThe 50-day moving average is 512.95.\n"

    assert [f.kind for f in lint_report(report, verified=verified)] == ["conflict"]


# --- presentation ----------------------------------------------------------


@pytest.mark.unit
def test_a_contradiction_leads_the_warning_block():
    report = (
        "The 50-day SMA is 111.05.\n\n"
        "Debt-to-equity is 49.00.\n\nDebt-to-equity stands at 100.\n"
    )

    block = render_warning_block(lint_report(report, verified=VERIFIED))
    kinds = [line for line in block.split("\n") if line.startswith("> **[")]

    assert kinds[0].startswith("> **[contradiction]")
    assert "names the correct value" in block


# --- end to end ------------------------------------------------------------


@pytest.mark.unit
def test_a_save_adjudicates_against_the_block_on_its_own_state(tmp_path):
    state = {
        "market_report": "The 50-day SMA is 111.05 and price sits beneath it.",
        "investment_debate_state": {"judge_decision": "**Rating**: Hold"},
        "risk_debate_state": {"judge_decision": "**Rating**: Hold"},
        "verified_market_block": render_trade_reference_block(
            _reference(), include_proposal_rule=False
        ),
    }

    paths = write_report_bundle(state, "INTC", tmp_path / "INTC_20260807_230554")
    warnings = (paths.markdown.parent / "numeric_warnings.md").read_text(encoding="utf-8")

    assert "[contradiction]" in warnings
    assert "verified value is 110.6" in warnings


@pytest.mark.unit
def test_a_state_without_a_block_still_gets_linted(tmp_path):
    """Losing adjudication must not cost the checks that never needed it."""
    state = {
        "market_report": "The 50-day SMA is 514.33.\n\nThe 50-day SMA is 512.95.",
        "investment_debate_state": {"judge_decision": "**Rating**: Hold"},
        "risk_debate_state": {"judge_decision": "**Rating**: Hold"},
    }

    paths = write_report_bundle(state, "INTC", tmp_path / "INTC_20260807_230554")
    warnings = Path(paths.markdown.parent / "numeric_warnings.md").read_text(encoding="utf-8")

    assert "[conflict]" in warnings
