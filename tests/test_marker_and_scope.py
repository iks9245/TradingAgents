"""The transaction marker must survive a fallback; a figure must keep its scope.

Two gaps left by earlier rounds. The ``FINAL TRANSACTION PROPOSAL`` line is
emitted by the render function, which does not run when the structured path
falls back — so the 2026-08-07 INTC report carried no transaction marker at all.
And Intel's 10-Q reported *server-product* ASP up 48% on a richer mix, which
reached the report as "Intel CPU prices up 48%" and then as evidence of
company-wide pricing power.
"""

from __future__ import annotations

import pytest

from tradingagents.agents.trader.trader import _restore_transaction_marker
from tradingagents.agents.utils.evidence_policy import (
    SCOPE_DISCIPLINE,
    get_evidence_discipline_instruction,
)
from tradingagents.agents.utils.rating import parse_rating, parse_trader_action
from tradingagents.agents.utils.structured import UNVALIDATED_MARKER

NOTICE = f"> ⚠️ {UNVALIDATED_MARKER} Trader's structured proposal could not be produced.\n\n"


@pytest.mark.unit
class TestTransactionMarkerSurvivesFallback:
    def test_an_unambiguous_action_is_restored(self):
        plan = NOTICE + "**Recommendation: Sell (Trim Long Position)**\n\nReduce exposure."
        assert "FINAL TRANSACTION PROPOSAL: **SELL**" in _restore_transaction_marker(plan)

    def test_an_ambiguous_text_is_not_guessed(self):
        # Two different labelled actions: the text does not say, so neither do we.
        plan = NOTICE + "Action: Buy\n\nOn reflection, Recommendation: Sell."
        out = _restore_transaction_marker(plan)
        assert "**UNDETERMINED**" in out
        assert "does not state one action unambiguously" in out

    def test_prose_with_no_labelled_action_is_not_guessed(self):
        plan = NOTICE + "The risk/reward is unattractive and the position is too large."
        assert "**UNDETERMINED**" in _restore_transaction_marker(plan)

    def test_a_validated_proposal_is_left_alone(self):
        # No fallback notice: the render function already emitted the marker.
        plan = "**Action**: Sell\n\nFINAL TRANSACTION PROPOSAL: **SELL**"
        assert _restore_transaction_marker(plan) == plan

    def test_an_existing_marker_is_never_duplicated(self):
        plan = NOTICE + "Action: Sell\n\nFINAL TRANSACTION PROPOSAL: **SELL**"
        assert _restore_transaction_marker(plan).count("FINAL TRANSACTION PROPOSAL") == 1

    def test_the_notice_itself_is_not_read_as_the_action(self):
        # The notice can quote a schema error naming an action word.
        plan = (
            "> ⚠️ **Unvalidated output.** Reason: Input should be 'Buy', 'Hold' or 'Sell'\n\n"
            "Recommendation: Sell\n"
        )
        assert parse_trader_action(plan) == "Sell"


@pytest.mark.unit
class TestRatingIgnoresCommentary:
    def test_a_quoted_notice_cannot_change_the_parsed_rating(self):
        # A schema rejection quotes the allowed values, and the notice is
        # prepended — so it would otherwise be read before the decision.
        decision = (
            "> ⚠️ **Unvalidated output.** Reason: rating: Input should be 'Buy'\n"
            ">\n"
            "**Rating**: Sell\n\nExit ahead of guidance."
        )
        assert parse_rating(decision) == "Sell"

    def test_a_lint_warning_block_is_ignored(self):
        decision = "> **[conflict]** Buy-side estimates differ\n\n**Rating**: Underweight"
        assert parse_rating(decision) == "Underweight"

    def test_an_ordinary_decision_still_parses(self):
        assert parse_rating("**Rating**: Overweight\n\nBuild slowly.") == "Overweight"


@pytest.mark.unit
class TestScopeDiscipline:
    def test_the_rule_names_the_boundaries_that_travel_with_a_figure(self):
        for clause in ("Which part of the business", "What drove it", "comparison basis"):
            assert clause in SCOPE_DISCIPLINE

    def test_it_forbids_the_specific_widening_that_shipped(self):
        assert '"Server product ASP" is not "CPU prices"' in SCOPE_DISCIPLINE
        assert "never restate a\nmix-driven change as a pricing action" in SCOPE_DISCIPLINE

    def test_downstream_agents_receive_it_alongside_the_evidence_tiers(self):
        instruction = get_evidence_discipline_instruction()
        assert "Evidence discipline" in instruction
        assert "Scope discipline" in instruction

    def test_the_news_analyst_carries_it_where_scope_is_first_lost(self):
        import inspect

        import tradingagents.agents.analysts.news_analyst as na

        assert "SCOPE_DISCIPLINE" in inspect.getsource(na)
