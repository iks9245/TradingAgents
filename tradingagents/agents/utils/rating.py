"""Shared 5-tier rating vocabulary and a deterministic heuristic parser.

The same five-tier scale (Buy, Overweight, Hold, Underweight, Sell) is used by:
- The Research Manager (investment plan recommendation)
- The Portfolio Manager (final position decision)
- The signal processor (rating extracted for downstream consumers)
- The memory log (rating tag stored alongside each decision entry)

Centralising it here avoids drift between those call sites.
"""

from __future__ import annotations

import re

# Canonical, ordered 5-tier scale (most bullish to most bearish).
RATINGS_5_TIER: tuple[str, ...] = (
    "Buy", "Overweight", "Hold", "Underweight", "Sell",
)

_RATING_SET = {r.lower() for r in RATINGS_5_TIER}

# Matches "Rating: X" / "rating - X" / "Rating: **X**" — tolerates markdown
# bold wrappers and either a colon or hyphen separator.
_RATING_LABEL_RE = re.compile(r"rating.*?[:\-][\s*]*(\w+)", re.IGNORECASE)


TRADER_ACTIONS: tuple[str, ...] = ("Buy", "Hold", "Sell")

_ACTION_SET = {a.lower() for a in TRADER_ACTIONS}

# "Action: Sell", "Recommendation: **Buy**", "FINAL TRANSACTION PROPOSAL: **HOLD**".
_ACTION_LABEL_RE = re.compile(
    r"(?:final transaction proposal|action|recommendation)\s*[:\-][\s*]*(\w+)",
    re.IGNORECASE,
)


def parse_trader_action(text: str) -> str | None:
    """Extract a Buy/Hold/Sell action from a trader's prose, or None if unclear.

    Deliberately refuses to guess. This runs only when the structured path fell
    back and the rendered ``FINAL TRANSACTION PROPOSAL`` line was lost with it —
    inferring a direction from ambiguous prose would put a fabricated
    instruction where a validated one used to be. Two different labelled actions
    in one text means the text does not say, so neither does this.
    """
    found: list[str] = []
    for line in text.splitlines():
        if line.lstrip().startswith(">"):
            continue  # commentary about the output, not the proposal
        for match in _ACTION_LABEL_RE.finditer(line):
            word = match.group(1).lower()
            if word in _ACTION_SET:
                found.append(word)
    if not found or len(set(found)) != 1:
        return None
    return found[0].capitalize()


def parse_rating(text: str, default: str = "Hold") -> str:
    """Heuristically extract a 5-tier rating from prose text.

    Two-pass strategy:
    1. Look for an explicit "Rating: X" label (tolerant of markdown bold).
    2. Fall back to the first 5-tier rating word found anywhere in the text.

    Returns a Title-cased rating string, or ``default`` if no rating word appears.
    """
    # Blockquoted lines are commentary *about* the output — the unvalidated-
    # fallback notice, a lint warning — not the decision itself. Their text can
    # name rating words (a schema rejection quotes the allowed values verbatim),
    # and since the notice is prepended it would otherwise be read first.
    lines = [line for line in text.splitlines() if not line.lstrip().startswith(">")]

    for line in lines:
        m = _RATING_LABEL_RE.search(line)
        if m and m.group(1).lower() in _RATING_SET:
            return m.group(1).capitalize()

    for line in lines:
        for word in line.lower().split():
            clean = word.strip("*:.,")
            if clean in _RATING_SET:
                return clean.capitalize()

    return default
