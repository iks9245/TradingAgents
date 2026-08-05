"""Tests for unit-explicit rendering of vendor fundamental fields (P0-1).

The regression these guard: yfinance's ``info`` holds fractions, percents, and
bare multiples in one flat namespace. A shipped report quoted ``debtToEquity``
6.01 — which is 6.01 PERCENT — as "6.01x, relatively high", inverting the
balance-sheet conclusion by a factor of 100.
"""

from __future__ import annotations

import pytest

from tradingagents.dataflows import fundamental_units as units


@pytest.mark.unit
class TestUnitRendering:
    def test_debt_to_equity_shows_percent_and_multiple(self):
        rendered = units.fmt_percent_with_multiple(6.01, "debtToEquity")
        assert "6.01%" in rendered
        assert "0.0601x" in rendered
        # The warning is the point: without it the bare number reads as a multiple.
        assert "PERCENT" in rendered
        assert "not a multiple" in rendered.lower()

    def test_margin_fraction_is_rendered_as_percent_with_raw_kept(self):
        rendered = units.fmt_fraction_as_pct(0.1440, "operatingMargins")
        assert "14.40%" in rendered
        assert "0.1440" in rendered
        assert "operatingMargins" in rendered

    def test_multiple_and_percent_never_render_the_same_way(self):
        # 2.73 as a current ratio and 2.73 as a debt/equity percent must not
        # produce interchangeable text.
        assert units.fmt_multiple(2.73) != units.fmt_percent_with_multiple(2.73)
        assert "2.73x" in units.fmt_multiple(2.73)

    def test_ambiguous_field_is_not_converted(self):
        rendered = units.fmt_raw(0.44, "dividendYield")
        assert "0.44" in rendered
        assert "do not convert" in rendered.lower()
        # No percentage is asserted, because the convention is not known.
        assert "44.00%" not in rendered

    def test_money_shows_magnitude_and_exact_value(self):
        rendered = units.fmt_money(845_600_000_000, "USD")
        assert "845.60B USD" in rendered
        assert "845,600,000,000" in rendered

    @pytest.mark.parametrize("value", [None, float("nan"), "n/a", "", True])
    def test_unusable_values_render_as_none(self, value):
        assert units.fmt_multiple(value) is None
        assert units.fmt_fraction_as_pct(value) is None
        assert units.fmt_money(value) is None


@pytest.mark.unit
class TestArithmeticHelpers:
    def test_pct_change_computes_growth(self):
        assert units.pct_change(346.39, 257.85) == pytest.approx(34.34, abs=0.01)

    @pytest.mark.parametrize("base", [0, -10.0])
    def test_pct_change_refuses_non_positive_base(self, base):
        # A swing from a negative or zero base has no meaningful percent change;
        # returning a number there would read as growth that did not happen.
        assert units.pct_change(50, base) is None

    def test_safe_ratio_matches_the_report_that_was_wrong(self):
        # The shipped report printed 14.4% for this division; it is 10.66%.
        assert units.safe_ratio(36.94, 346.39) == pytest.approx(0.1066, abs=0.0001)
        # ...and 13.4% for this one; it is 12.51%.
        assert units.safe_ratio(43.35, 346.39) == pytest.approx(0.1251, abs=0.0001)

    def test_safe_ratio_rejects_zero_denominator(self):
        assert units.safe_ratio(1.0, 0) is None

    def test_render_field_dispatch_rejects_unknown_kind(self):
        with pytest.raises(KeyError):
            units.render_field("percentage-ish", 1.0)
