"""Provenance rules that keep an unverified claim from becoming a stated fact.

The sentiment analyst quotes retail posts by design — that is its job. The
failure is what happens downstream. In one shipped report, "capex doubled and
free cash flow fell 39%" originated in a StockTwits post; the sentiment
analyst relayed it correctly as retail opinion, and from there the bear
researcher, the research manager, and the portfolio manager each restated it
as established fact. It became a pillar of an Underweight rating while the
same report's own cash-flow table showed free cash flow up 180% year on year
and capex up 53%.

Nothing in the pipeline distinguished "a number computed from a filing" from
"a number somebody typed on a message board". These constants add that
distinction: one marking rule for the agent that handles social sources, and
one carrying rule for every agent that reads its output.
"""

from __future__ import annotations

# Marker the sentiment analyst wraps around numbers that came from social posts.
UNVERIFIED_MARKER = "[UNVERIFIED — social post]"

# For the sentiment analyst: how to mark quantitative claims it relays.
SOURCE_MARKING_INSTRUCTION = f"""
## Marking unverified numbers

StockTwits and Reddit posts are opinion, not filings. Retail posters routinely
state financial figures that are wrong, stale, or invented.

Whenever you relay a **quantitative claim** that originates from a StockTwits or
Reddit post — a growth rate, a margin, a multiple, a cash-flow figure, a price
target, a valuation — you must:

1. Wrap it in the marker `{UNVERIFIED_MARKER}` at the point of use, e.g.
   "posters cite a 39% drop in free cash flow {UNVERIFIED_MARKER}".
2. Attribute it to the platform, never to the company or to an analyst.
3. Never present it as a fact, and never use it to support your own conclusion.

News headlines are a middle tier: attribute the figure to the outlet by name and
say it is as reported, not as verified. Figures from the fundamentals or market
snapshot are verified and need no marker.

List every marked claim in the `unverified_numeric_claims` field so downstream
agents can see them without re-parsing your prose. If a social post's number
contradicts the fundamentals or market data, say so explicitly rather than
picking one.
""".strip()

# A figure's scope is part of the figure. Intel's 10-Q reported server-product
# ASP up 48% year on year, driven mainly by a richer product mix. That reached a
# shipped report as "Intel CPU prices up 48%", and from there as evidence of
# pricing power across the whole CPU business — a claim the source does not
# make. Nothing was misquoted: the number survived and its scope did not.
SCOPE_DISCIPLINE = """
**Scope discipline.** A quantitative claim carries the boundary it was measured
within, and that boundary travels with the number. When you relay a figure, keep:

- **Which part of the business** it covers — a segment, a product line, a region.
  "Server product ASP" is not "CPU prices"; a data-centre figure is not a company
  figure.
- **What drove it**, when the source says. A price rise driven by a shift toward
  higher-end products is a mix effect, not a price increase to customers, and the
  two support different conclusions about pricing power.
- **The comparison basis** — year on year, quarter on quarter, sequential.

Widening a figure's scope is not summarising, it is a different and stronger
claim than the source made. If you cannot state the boundary, say the figure is
scoped to something you could not determine rather than dropping the qualifier.
Never generalise from a segment to the whole company, and never restate a
mix-driven change as a pricing action.
""".strip()

# For every agent downstream of the sentiment analyst.
EVIDENCE_DISCIPLINE = f"""
**Evidence discipline.** The reports you were given mix three tiers of evidence,
and they do not carry equal weight:

1. **Verified** — figures from the verified market snapshot or the verified
   fundamentals snapshot. These are computed from price data and filings. Cite
   them freely, with the period the snapshot gave them.
2. **Reported** — figures attributed to a named news outlet or analyst. Cite them
   with the attribution attached ("as reported by X"), never as your own finding.
3. **Unverified** — anything marked `{UNVERIFIED_MARKER}`, or any number whose
   only source is a StockTwits or Reddit post. You may cite these ONLY as a
   description of what market participants believe — "retail posters are worried
   that ..." — never as a fact and never as support for your recommendation. Keep
   the marker attached when you repeat one.

If an unverified number contradicts a verified one, the verified figure stands;
name the conflict rather than averaging or reconciling them. Do not upgrade a
claim's tier by restating it: a number that arrived unverified stays unverified
no matter how many times it has been repeated in the debate history.
""".strip()


def get_evidence_discipline_instruction() -> str:
    """Return the downstream evidence-tier and scope rules, ready for prompt assembly."""
    return "\n\n" + EVIDENCE_DISCIPLINE + "\n\n" + SCOPE_DISCIPLINE
