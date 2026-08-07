"""A schema rejection must be corrected or disclosed, never silently bypassed.

On the 2026-08-07 INTC run the trader's structured call fell back to free text,
and the prose that shipped placed a stop-loss *above* the entry price of a long
— exactly what ``TraderProposal`` rejects. The guard fired and the fallback
published the unvalidated version anyway, in a section that read like every
validated one. A silently-bypassed check is worse than no check.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tradingagents.agents.schemas import (
    PositionIntent,
    TraderAction,
    TraderProposal,
    render_trader_proposal,
)
from tradingagents.agents.utils.structured import invoke_structured_or_freetext

VALID = TraderProposal(
    action=TraderAction.SELL,
    position_intent=PositionIntent.REDUCE_LONG,
    reasoning="Trim into strength.",
    entry_price=99.81,
    stop_loss=87.39,
)


def _rejection() -> Exception:
    """The real error the schema raises for the shipped proposal."""
    try:
        TraderProposal(
            action=TraderAction.SELL,
            position_intent=PositionIntent.REDUCE_LONG,
            reasoning="x", entry_price=99.81, stop_loss=112.23,
        )
    except Exception as exc:  # noqa: BLE001 — this is the object under test
        return exc
    raise AssertionError("schema should have rejected a long stop above entry")


@pytest.mark.unit
class TestValidationFailureIsRetried:
    def test_the_model_gets_one_chance_to_fix_its_own_error(self):
        seen = []

        def invoke(prompt):
            seen.append(prompt)
            if len(seen) == 1:
                raise _rejection()
            return VALID

        structured = MagicMock()

        structured.invoke.side_effect = invoke
        plain = MagicMock()

        out = invoke_structured_or_freetext(
            structured, plain, "base", lambda r: render_trader_proposal(r), "Trader"
        )
        assert len(seen) == 2
        assert "**Stop Loss**: 87.39" in out
        plain.invoke.assert_not_called()

    def test_the_retry_carries_the_specific_defect(self):
        seen = []

        def invoke(prompt):
            seen.append(prompt)
            if len(seen) == 1:
                raise _rejection()
            return VALID

        structured = MagicMock()

        structured.invoke.side_effect = invoke
        invoke_structured_or_freetext(structured, MagicMock(), "base", lambda r: "ok", "Trader")

        retry = seen[1]
        assert "must be BELOW entry_price" in retry
        assert "not a formatting problem" in retry
        assert retry.startswith("base")

    @pytest.mark.parametrize(
        "prompt,kind",
        [
            ("plain string", str),
            ([{"role": "system", "content": "s"}, {"role": "user", "content": "u"}], list),
        ],
    )
    def test_the_correction_matches_the_prompt_shape(self, prompt, kind):
        seen = []

        def invoke(p):
            seen.append(p)
            if len(seen) == 1:
                raise _rejection()
            return VALID

        structured = MagicMock()

        structured.invoke.side_effect = invoke
        invoke_structured_or_freetext(structured, MagicMock(), prompt, lambda r: "ok", "Trader")
        assert isinstance(seen[1], kind)
        if kind is list:
            assert len(seen[1]) == len(prompt) + 1
            assert seen[1][-1]["role"] == "user"


@pytest.mark.unit
class TestUnvalidatedOutputIsDisclosed:
    def _fallback(self, error: Exception, content: str = "prose") -> str:
        structured = MagicMock()
        structured.invoke.side_effect = error
        plain = MagicMock()
        plain.invoke.return_value = MagicMock(content=content)
        return invoke_structured_or_freetext(
            structured, plain, "p", lambda r: "x", "Trader",
            bypassed_checks="the stop-loss direction check",
        )

    def test_persistent_rejection_is_labelled(self):
        out = self._fallback(_rejection(), "Stop-loss above entry at $112.23")
        assert out.startswith("> ⚠️ **Unvalidated output.**")
        assert "the stop-loss direction check" in out
        assert "unchecked" in out
        assert "Stop-loss above entry at $112.23" in out

    def test_the_notice_stays_inside_its_blockquote(self):
        # A pydantic error spans several lines; left raw it would break out of
        # the quote and bury the warning in stack-trace furniture.
        out = self._fallback(_rejection())
        notice = out.split("\n\n", 1)[0]
        assert all(line.startswith(">") for line in notice.split("\n"))

    def test_a_provider_without_structured_output_is_not_retried(self):
        plain = MagicMock()
        plain.invoke.return_value = MagicMock(content="prose")
        out = invoke_structured_or_freetext(
            None, plain, "p", lambda r: "x", "PM", bypassed_checks="the rating check"
        )
        assert "unsupported by this provider" in out
        plain.invoke.assert_called_once()

    def test_a_non_validation_error_is_not_retried(self):
        # A transport failure is not something the model can fix by trying
        # again with the message quoted back at it.
        structured = MagicMock()
        structured.invoke.side_effect = ConnectionError("socket closed")
        plain = MagicMock()
        plain.invoke.return_value = MagicMock(content="prose")
        invoke_structured_or_freetext(structured, plain, "p", lambda r: "x", "Trader")
        assert structured.invoke.call_count == 1

    def test_a_successful_call_carries_no_notice(self):
        structured = MagicMock()
        structured.invoke.return_value = VALID
        out = invoke_structured_or_freetext(
            structured, MagicMock(), "p", render_trader_proposal, "Trader"
        )
        assert "Unvalidated output" not in out
