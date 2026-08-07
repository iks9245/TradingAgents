"""Shared helpers for invoking an agent with structured output and a graceful fallback.

The Portfolio Manager, Trader, and Research Manager all follow the same
canonical pattern:

1. At agent creation, wrap the LLM with ``with_structured_output(Schema)``
   so the model returns a typed Pydantic instance. If the provider does
   not support structured output (rare; mostly older Ollama models), the
   wrap is skipped and the agent uses free-text generation instead.
2. At invocation, run the structured call and render the result back to
   markdown. If the structured call itself fails for any reason
   (malformed JSON from a weak model, transient provider issue), fall
   back to a plain ``llm.invoke`` so the pipeline never blocks.

Centralising the pattern here keeps the agent factories small and ensures
all three agents log the same warnings when fallback fires.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, TypeVar

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Schema-only structured output binds exactly one tool (the schema itself), so a
# model that reaches for a search tool emits an unknown tool call and the whole
# structured attempt is discarded for a free-text retry. Agents on this path
# state the constraint explicitly rather than relying on the binding alone
# (#1130).
NO_EXTERNAL_TOOLS = (
    "Use only the evidence provided in this prompt. Do not call external tools "
    "or search the web; if something is missing, say so explicitly."
)


def bind_structured(llm: Any, schema: type[T], agent_name: str) -> Any | None:
    """Return ``llm.with_structured_output(schema)`` or ``None`` if unsupported.

    Logs a warning when the binding fails so the user understands the agent
    will use free-text generation for every call instead of one-shot fallback.
    """
    try:
        return llm.with_structured_output(schema)
    except (NotImplementedError, AttributeError) as exc:
        logger.warning(
            "%s: provider does not support with_structured_output (%s); "
            "falling back to free-text generation",
            agent_name, exc,
        )
        return None


def _is_validation_failure(exc: BaseException) -> bool:
    """True when the schema rejected the model's answer, rather than the call failing.

    The distinction decides what to do next. A provider that cannot produce
    structured output at all is a capability problem, and retrying is pointless.
    A schema rejection is the model getting something wrong that it can fix when
    told what — and it is the case that matters most, because the rejection
    message describes a real defect in the proposal.
    """
    if isinstance(exc, ValidationError):
        return True
    text = str(exc).lower()
    return "validation error" in text or "value error" in text


def _with_correction(prompt: Any, exc: BaseException) -> Any:
    """Return ``prompt`` with the schema's complaint appended, in its own shape."""
    correction = (
        "Your previous answer was rejected because it failed validation:\n\n"
        f"{exc}\n\n"
        "This is a real inconsistency in the proposal, not a formatting problem. "
        "Fix the underlying values — do not restate the same numbers — and answer again."
    )
    if isinstance(prompt, str):
        return prompt + "\n\n" + correction
    if isinstance(prompt, list):
        messages = list(prompt)
        if messages and isinstance(messages[0], dict):
            return messages + [{"role": "user", "content": correction}]
        return messages + [HumanMessage(content=correction)]
    return prompt


# Callers detect a fallback by looking for this in the returned text.
UNVALIDATED_MARKER = "**Unvalidated output.**"


def _mark_unvalidated(text: str, agent_name: str, bypassed_checks: str, reason: str) -> str:
    """Prefix free-text output with a notice that no structured checks ran.

    A silently-bypassed guard is worse than no guard: the report looks exactly
    as it does when every check passed. On the 2026-08-07 INTC run the trader's
    structured call fell back, and the resulting prose placed a stop-loss
    *above* the entry price of a long — precisely what the schema rejects —
    while reading like a validated proposal.
    """
    checks = bypassed_checks or "the schema's consistency checks"
    # A pydantic error spans several lines; left raw it would break out of the
    # blockquote and bury the notice in stack-trace furniture.
    condensed = " ".join(str(reason).split())
    if len(condensed) > 200:
        condensed = condensed[:197] + "..."
    return (
        f"> ⚠️ {UNVALIDATED_MARKER} {agent_name}'s structured proposal could not be "
        f"produced, so this section is free-form text that bypassed {checks}. "
        f"Treat its levels and internal consistency as unchecked.\n"
        f">\n"
        f"> Reason: {condensed}\n\n"
        f"{text}"
    )


def invoke_structured_or_freetext(
    structured_llm: Any | None,
    plain_llm: Any,
    prompt: Any,
    render: Callable[[T], str],
    agent_name: str,
    bypassed_checks: str = "",
) -> str:
    """Run the structured call and render to markdown, falling back to free text.

    ``prompt`` is whatever the underlying LLM accepts (a string for chat
    invocations, a list of message dicts for chat models that take that
    shape). The same value is forwarded to the free-text path so the
    fallback sees the same input the structured call did.

    A schema rejection earns one retry with the complaint fed back, because the
    rejection names a concrete defect the model can correct. Anything that
    reaches the free-text path is labelled as unvalidated in the output.
    """
    reason = "structured output unsupported by this provider"

    if structured_llm is not None:
        result, error = _attempt_structured(structured_llm, prompt, agent_name)
        if result is not None:
            return render(result)
        reason = str(error) if error else "no parsed result"

        if error is not None and _is_validation_failure(error):
            logger.warning(
                "%s: structured output failed validation (%s); retrying once with the "
                "error fed back", agent_name, error,
            )
            retried, retry_error = _attempt_structured(
                structured_llm, _with_correction(prompt, error), agent_name
            )
            if retried is not None:
                return render(retried)
            reason = f"failed validation twice: {retry_error or error}"

        logger.warning(
            "%s: falling back to unvalidated free text (%s)", agent_name, reason,
        )

    response = plain_llm.invoke(prompt)
    return _mark_unvalidated(response.content, agent_name, bypassed_checks, reason)


def _attempt_structured(
    structured_llm: Any, prompt: Any, agent_name: str
) -> tuple[T | None, BaseException | None]:
    """One structured call: return the parsed result, or the failure that stopped it."""
    try:
        result = structured_llm.invoke(prompt)
        if result is None:
            # A thinking model can answer in plain text instead of calling the
            # tool, leaving the parser with nothing to return.
            return None, ValueError("structured output returned no parsed result")
        return result, None
    except Exception as exc:  # noqa: BLE001 — every failure mode routes to the caller
        logger.debug("%s: structured attempt failed: %s", agent_name, exc)
        return None, exc
