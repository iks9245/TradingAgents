"""Adjudicating a report's fundamental ratios against the run's own snapshot.

The market block joined Gate 3's adjudication because one shared table drives
its renderer and its parser. The fundamentals snapshot did not — not because its
figures are less checkable, but because every section here computes a value and
formats it in the same breath, leaving prose as the only thing downstream could
read. These tests pin the retained figures, and the one rule that makes them
usable: a ratio is honestly written two ways, so the reading's own marker
decides which convention it is checked in.
"""

import re
from dataclasses import fields

import pandas as pd
import pytest

import tradingagents.dataflows.fundamentals_validator as validator
from tests.test_fundamentals_validator import _ticker
from tradingagents.agents.utils.verified_evidence import (
    VerifiedEvidence,
    verified_figures_from_state,
)
from tradingagents.dataflows.fundamentals_validator import (
    _FACT_METRICS,
    FundamentalFacts,
)
from tradingagents.report_lint import (
    _METRIC_ALIASES,
    _RATIO_METRICS,
    _metric_values,
    lint_report,
)

# 6.01% leverage, the figure whose two readings are the whole problem.
FACTS = FundamentalFacts(debt_to_equity=0.060134, current_ratio=2.73)


def _kinds(findings) -> list[str]:
    return [finding.kind for finding in findings]


# --- the retained record -----------------------------------------------------


@pytest.mark.unit
def test_every_mapped_field_exists_on_the_record():
    """The table is the contract; a typo in it would silently drop a figure."""
    known = {field.name for field in fields(FundamentalFacts)}
    for _metric, field_name in _FACT_METRICS:
        assert field_name in known


@pytest.mark.unit
def test_every_mapped_metric_is_one_the_linter_knows():
    """A figure keyed to a metric with no alias is retained for nobody."""
    for metric, _field in _FACT_METRICS:
        assert metric in _METRIC_ALIASES


@pytest.mark.unit
def test_absent_figures_are_omitted_rather_than_zeroed():
    """A missing ratio must fall back to the weaker check, not be adjudicated.

    Zero is a real leverage reading, so a defaulted 0.0 would contradict every
    correct statement in the report.
    """
    assert FundamentalFacts().verified_figures() == {}
    assert FundamentalFacts(current_ratio=2.73).verified_figures() == {
        "current_ratio": 2.73
    }


@pytest.mark.unit
def test_the_record_is_dimensionless():
    """The stored convention is the ratio, which the consumer scales from."""
    assert FACTS.verified_figures()["debt_to_equity"] == pytest.approx(0.060134)


# --- a ratio has two honest forms --------------------------------------------


@pytest.mark.unit
def test_both_forms_of_the_same_ratio_are_confirmed():
    """The regression this feature could most easily have shipped.

    The snapshot itself prints "6.01%  (= 0.0601x)", and a report quoting it
    faithfully states both. Adjudicating both against one unscaled number would
    report one of the two correct readings as a contradiction on every report.
    """
    report = (
        "Debt-to-equity of 6.01%. The balance-sheet table gives the same\n"
        "leverage as debt-to-equity of 0.0601x."
    )
    # Both forms must actually reach the check, or this pins nothing.
    assert len(_metric_values(report)["debt_to_equity"]) == 2
    assert lint_report(report, verified=FACTS.verified_figures()) == []


@pytest.mark.unit
def test_a_wrong_percent_reading_is_named():
    report = "Debt-to-equity of 61.5%, a heavy balance sheet."
    findings = lint_report(report, verified=FACTS.verified_figures())
    assert _kinds(findings) == ["contradiction"]
    assert "61.5" in findings[0].summary
    # The convention is named, or "stated as 61.5, verified 6.01" reads as
    # nonsense to someone who cannot see which units each side is in.
    assert "percent form" in findings[0].summary


@pytest.mark.unit
def test_a_wrong_multiple_reading_is_named():
    report = "Leverage is high: debt-to-equity of 1.85x."
    findings = lint_report(report, verified=FACTS.verified_figures())
    assert _kinds(findings) == ["contradiction"]
    assert "multiple form" in findings[0].summary


@pytest.mark.unit
def test_a_consistently_wrong_ratio_is_still_caught():
    """The shape with no spread, which is why adjudication exists at all."""
    report = (
        "Debt-to-equity of 61.5%. A debt-to-equity of 61.5% implies a stretched "
        "balance sheet, and debt-to-equity of 61.5% leaves little room."
    )
    assert _kinds(lint_report(report, verified=FACTS.verified_figures())) == [
        "contradiction"
    ]


@pytest.mark.unit
def test_rounding_to_the_precision_offered_is_faithful():
    report = "Debt-to-equity of 6%."
    assert lint_report(report, verified=FACTS.verified_figures()) == []


# --- what is deliberately not adjudicated ------------------------------------


@pytest.mark.unit
def test_a_bare_ratio_is_left_to_the_spread_check():
    """A bare "6.01" is ambiguous, and a contradiction names a value as wrong.

    Whether it is the 100x unit error this codebase has shipped once or a writer
    omitting a percent sign is not recoverable from the text, so the weaker
    finding — two readings disagree — is the honest one.
    """
    report = "Debt-to-equity of 6.01. Elsewhere the report gives debt-to-equity of 0.0601."
    findings = lint_report(report, verified=FACTS.verified_figures())
    assert _kinds(findings) == ["conflict"]


@pytest.mark.unit
def test_one_bare_ratio_alone_produces_nothing():
    """No spread and no adjudicable marker leaves nothing that can be asserted."""
    assert lint_report(
        "Debt-to-equity of 6.01.", verified=FACTS.verified_figures()
    ) == []


@pytest.mark.unit
def test_market_metrics_keep_adjudicating_bare_numbers():
    """The ratio rule must not leak: a price level has one honest form."""
    report = "The 50-day SMA is 99.43."
    assert _kinds(lint_report(report, verified={"sma50": 110.60})) == ["contradiction"]


@pytest.mark.unit
def test_the_ratio_rule_covers_exactly_the_ratio_metrics():
    """A new ratio metric added without a scale would be adjudicated unscaled."""
    assert {"debt_to_equity", "current_ratio"} == _RATIO_METRICS


# --- reaching the linter from a run ------------------------------------------


@pytest.mark.unit
def test_state_figures_carry_market_and_fundamentals_together():
    state = {
        "verified_market_block": "",
        "verified_fundamentals_figures": {"debt_to_equity": 0.060134},
    }
    assert verified_figures_from_state(state) == {"debt_to_equity": 0.060134}


@pytest.mark.unit
def test_a_state_without_figures_still_yields_the_market_ones():
    """Older runs and bare programmatic states must not lose what they do have."""
    assert verified_figures_from_state({"verified_market_block": ""}) == {}


@pytest.mark.unit
@pytest.mark.parametrize("junk", [None, "6.01", {"debt_to_equity": "6.01"}, {"x": True}])
def test_unusable_figure_payloads_are_dropped_not_coerced(junk):
    """A resumed checkpoint or hand-built state can carry anything at all."""
    state = {"verified_market_block": "", "verified_fundamentals_figures": junk}
    assert verified_figures_from_state(state) == {}


@pytest.mark.unit
def test_an_unavailable_snapshot_contributes_no_figures(monkeypatch):
    """Failing open must mean no adjudication, never adjudication against zero."""
    monkeypatch.setattr(
        validator,
        "build_verified_fundamentals",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("vendor down")),
    )
    validator.resolve_fundamentals_snapshot.cache_clear()

    snapshot = validator.resolve_fundamentals_snapshot("INTC", "2026-08-07")
    assert "UNAVAILABLE" in snapshot.block
    assert snapshot.facts.verified_figures() == {}


@pytest.mark.unit
def test_evidence_defaults_to_no_figures():
    """Constructing the record without figures must not fail a caller."""
    assert VerifiedEvidence("market", "fundamentals").figures == {}


# --- the retained figure and the rendered one are the same figure ------------


@pytest.mark.unit
class TestRetainedFiguresMatchTheBlock:
    """The property the market path gets from sharing a render/parse table.

    Here the record is carried rather than parsed back, so the risk is not a
    misread but a drift: a section changing what it prints while the retained
    value stays as it was. Reading the block's own rendered figure back and
    comparing it to the record is what would catch that.
    """

    def _snapshot(self, monkeypatch) -> validator.FundamentalsSnapshot:
        fake = _ticker()
        monkeypatch.setattr(validator.yf, "Ticker", lambda symbol: fake)
        monkeypatch.setattr(validator, "load_ohlcv", lambda symbol, date: pd.DataFrame())
        return validator.build_verified_fundamentals("AMD", "2025-12-31")

    def test_leverage_is_retained_dimensionless(self, monkeypatch):
        facts = self._snapshot(monkeypatch).facts
        # 3,871,000 / 64,462,000
        assert facts.debt_to_equity == pytest.approx(0.060051, abs=1e-6)

    def test_current_ratio_is_retained(self, monkeypatch):
        facts = self._snapshot(monkeypatch).facts
        # 26,834,000 / 9,829,000
        assert facts.current_ratio == pytest.approx(2.7301, abs=1e-4)

    def test_the_block_still_prints_both_readings(self, monkeypatch):
        """Retaining a figure must not change what the analyst is shown."""
        block = self._snapshot(monkeypatch).block
        assert "| Total debt / equity | 6.01%  (= 0.0601x)" in block

    def test_the_rendered_percent_agrees_with_the_record(self, monkeypatch):
        """Drift between what is printed and what is retained is the failure."""
        snapshot = self._snapshot(monkeypatch)
        rendered = re.search(
            r"\| Total debt / equity \| ([\d.]+)%", snapshot.block
        )
        assert rendered is not None
        assert float(rendered.group(1)) == pytest.approx(
            snapshot.facts.debt_to_equity * 100, abs=0.005
        )

    def test_a_faithful_report_of_this_snapshot_lints_clean(self, monkeypatch):
        """End to end: the block, a report quoting it, and no warning."""
        figures = self._snapshot(monkeypatch).facts.verified_figures()
        report = "Debt-to-equity of 6.01%, and a current ratio of 2.73x."
        assert lint_report(report, verified=figures) == []
