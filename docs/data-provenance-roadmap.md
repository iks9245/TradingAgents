# Where the verified data goes next

Mostly a plan, not a record. [`architecture.md`](architecture.md) describes what
runs today; this describes three pieces of work, why they are ordered the way
they are, and — for each — what it can honestly deliver versus what it cannot.

**Stage 1a has since landed.** It is kept here in full rather than deleted,
because the reasoning behind its boundaries is what the later stages inherit, and
because what it deliberately left undone is the specification for 1b. Everything
else below is still unbuilt.

The three gates now catch a wrong number in a finished report by comparing the
report against the run's own market snapshot. The work below extends that
reference in three directions: to the fundamentals figures the market block does
not cover, to a primary source that can adjudicate the vendor itself, and to the
news text a relayed figure came from.

## The shape of the problem

Gate 3 gained real adjudication when the market snapshot became readable back
out of its own rendered block. That was possible because of how the market path
is built, not because market figures are inherently more checkable:

```python
_LEVEL_LINES: tuple[tuple[str, str], ...] = (
    ("Last close", "close"),
    ("ATR (daily volatility)", "atr"),
    ...
)
```

One table of `(label, field)` pairs drives both `render_trade_reference_block`
and `parse_trade_reference_block`, and `TradeReference` is a frozen dataclass. A
level cannot be added to the block without the parser seeing it, and reading the
block back is exact rather than a guess at prose.

`fundamentals_validator.py` is built the opposite way. Every section is an
`_append_*_section(lines: list[str], ...)` that computes a value and formats it
in the same breath:

```python
def _margin_pct(numerator: float | None, denominator: float | None) -> float | None:
    ratio = safe_ratio(numerator, denominator)
    return None if ratio is None else ratio * 100
```

The float exists. It is rendered into a table cell, and then it is gone. Nothing
downstream — not the linter, not a consumer reading the report off disk — can
recover it except by parsing prose, which
`verified_evidence.verified_figures_from_state` declines to do for a stated
reason: the snapshot's shape depends on whichever statements a vendor returned,
and guessing figures out of it would put invented authority behind a warning.

So the constraint is not that fundamentals are unverifiable. It is that the
numbers are discarded at the moment they are formatted.

**What that costs, concretely.** The linter knows nine metrics. Five come from
the market block and are adjudicated against source. The remaining four —
`debt_to_equity`, `current_ratio`, `ocf`, `fcf` — are the ones with the worst
history in shipped reports:

| Metric | What went wrong | What caught it |
|---|---|---|
| `debt_to_equity` | 6.01 read as a ratio, not 6.01% — a 100× unit error that inverted the balance-sheet conclusion | A human reading the report |
| `ocf` / `fcf` | One figure quoted under both labels by three agents in turn, the last calling it verified | A human reading the report |

Both still get only the spread check, which compares the report against itself.
A figure that is wrong but stated consistently passes it untouched.

## Stage 1 — Retain the figures the snapshot already computes

The seam is smaller than it looks. `lint_report` is already metric-agnostic:

```python
def lint_report(markdown: str, *, verified: Mapping[str, float] | None = None) -> list[Finding]:
```

`_metric_findings` routes each metric to `_contradiction_findings` when a
verified figure exists and to `_spread_findings` when it does not. **Adding a
metric requires no change to the linter.** The only reason fundamentals figures
never arrive is that `verified_figures_from_state` returns
`parse_trade_reference_block(market)` and nothing else.

### 1a — Scalar metrics *(landed)*

Give the snapshot a typed record alongside its rendered block, following the
market path's pattern: one shared field table, a frozen dataclass, render and
access driven by the same source.

| Where | Change |
|---|---|
| `dataflows/fundamentals_validator.py` | A `FundamentalFacts` record; `_append_balance_section` returns the ratios it computed rather than only formatting them |
| `agents/utils/verified_evidence.py` | `resolve_verified_evidence` returns a `VerifiedEvidence` record carrying the figures beside the two blocks |
| `agents/utils/agent_states.py` | A `verified_fundamentals_figures` key beside the two existing block keys |
| `graph/trading_graph.py` | Store it where the blocks are already stored |
| `report_lint.py` | Unit-aware adjudication for ratio metrics |

A report stating `debt_to_equity` as `61.5%` against a snapshot holding `0.0601`
now produces a `[contradiction]` naming the correct value, rather than passing
silently.

Two things this plan predicted wrongly, recorded because the same guesses would
otherwise be repeated for 1b:

**`report_lint` needed more than tolerances, and no tolerances at all.** The 1%
`_DEFAULT_TOLERANCE` was already right for ratios, so `_METRIC_TOLERANCE` gained
no entries. What it did need was the rule below, which the plan did not
anticipate — "adding a metric requires no linter change" held for the market
metrics and not for these.

**A ratio has two honest forms.** `debt_to_equity` is in `_RATIO_METRICS`, where
percent- and multiple-marked readings are both kept, and the snapshot prints both
on purpose. Adjudicating both against one unscaled figure would have flagged one
of the two *correct* forms on every report quoting the snapshot faithfully — a
guaranteed false positive, discovered only when the code was written. The stored
figure is dimensionless and the statement's own marker decides what it is
compared to. See `architecture.md`, "A ratio has two honest forms".

Also unchanged from the plan, and worth keeping in view for 1b: block and figures
resolve and cache together, since they were already computed in one pass; and an
unavailable snapshot contributes no figures rather than zeroed ones, because zero
is a real leverage reading.

### 1b — Series metrics

`ocf` and `fcf` carry one value per period, which is why `_SERIES_ALIASES` keeps
them out of the conflict check: five quarters of operating cash flow are five
facts, not a self-contradiction. Adjudicating them needs an interface wider than
`Mapping[str, float]` — a metric maps to the period series, not to one number.

The return justifies the widening. The cross-label check currently says that one
figure appears under two incompatible labels and that one of them was read from
the wrong column. With the real per-period values in hand it can say **which**:
*−2.54B is the Q1 free cash flow, not operating cash flow.* That is the exact
defect three agents propagated, and naming the wrong column is the difference
between a warning a reader must investigate and one they can act on.

### What stage 1 deliberately leaves out

**Period-bearing ratios stay on the spread check.** A margin means nothing
without its period, and the linter's metric vocabulary has no period concept.
Adjudicating a period-bearing figure against a single scalar would fire whenever
a report legitimately discusses an earlier quarter — manufacturing exactly the
false-positive class that two rounds of lint fixes were spent eliminating. A
warning block that cries wolf is one readers learn to skip, and that failure is
more expensive than the coverage it would buy.

`debt_to_equity` and `current_ratio` are point-in-time balance-sheet values with
no period ambiguity, which is why the first cut is drawn there. Extending to
margins means giving metric keys a period, and that is its own piece of work with
its own false-positive budget — not a widening of this one.

## Stage 2 — A primary source under the vendor

Every fundamentals cross-check today compares a vendor against itself.
`_append_operating_income_crosscheck` states its own ceiling plainly:

> This detects internal inconsistency only. When the vendor's own expense row is
> itself understated, no arithmetic over its numbers can recover the filed
> figure.

The comparison is also one-sided on purpose: a recomputed figure *below* the
reported one means the vendor booked less expense than it itemised, which is its
defect; *above* only means this code does not know every expense row in use, and
flagging that would blame a vendor for a gap in our own enumeration. Both
choices are right, and both are bounded by the same thing — the check has
nothing to look at except the vendor. A period it marks `consistent` is not
thereby confirmed.

SEC EDGAR's `companyfacts` API gives it something: XBRL-tagged values, no API
key, and an accession number per fact that connects directly to the existing
provenance stamp.

Three existing contracts constrain how it lands:

1. **Look-ahead filtering keys on the filing date, not the period end.** A
   quarter ending 30 June is filed weeks later; filtering on the period end
   feeds a July backtest figures that were not public until August. This is the
   shape of #1115 with a different vendor attached, and it is the one mistake
   here that silently corrupts historical results rather than failing loudly.
2. **EDGAR joins the `data_vendors` chain; it is not added silently.** The
   configured vendor list is the exact resolution chain — a request must never
   be routed to a vendor the user did not choose.
3. **US registrants only.** `0700.HK` and its neighbours still resolve through
   the existing vendors, so the one-sided cross-check stays as the fallback and
   is upgraded to two-sided only where a filing is actually available.

This is also the only route to the gap `87634ce` documented and could not close:
a company-defined free cash flow exists, comes from the filings, is unavailable
from this vendor, and can carry the opposite sign — Intel's +4,450M simplified
against −8,419M as the company reports it. Today that gap is disclosed. With
filings in reach it can be resolved.

## Stage 3 — Keep the source text a relayed figure came from

`architecture.md` lists scope discipline as a rule that cannot be enforced
deterministically, "no source text to diff against". That is true of the code as
written, but it describes a data-layer choice rather than a fixed limit: the
article text is in hand when it is fetched, and it is simply not kept.

What keeping it buys is worth stating precisely, because it is less than it
first appears.

**Deterministically checkable:** whether a figure attributed to news appears in
any retrieved source at all. That catches a number with no provenance.

**Not deterministically checkable:** scope widening itself. Distinguishing
"*server-product* ASP up 48%, driven by product mix" from "Intel CPU prices up
48%" is a semantic comparison. The three gates deliberately ask an LLM nothing,
and the alternative within that constraint is keyword-overlap heuristics — the
category this codebase is right to be wary of, since a warning block's precision
matters more than its reach.

So stage 3's deliverable is **traceability, not enforcement**: every relayed
figure carries a link back to the source sentence, so the check a person
performs takes one click instead of a search. That follows the rule the rest of
the codebase already applies to gaps it cannot close — disclose it, rather than
resolve it quietly in whichever direction suits.

Whether a heuristic check belongs on top is a question to answer after the
sources are retained and can be tried against real cases, not before.

## Sequencing

| # | Work | Depends on | Size | State |
|---|---|---|---|---|
| 1 | `FundamentalFacts`; scalar metrics reach adjudication | — | Medium | **Landed** |
| 2 | Series interface; cross-label names the wrong column | 1 | Medium | Next |
| 3 | `dataflows/sec_edgar.py`; look-ahead by filing date; vendor chain | — | Large | |
| 4 | Two-sided operating-income check; as-filed FCF | 2, 3 | Medium | |
| 5 | News source retention and figure traceability | — | Medium | |

1 and 3 are independent — the EDGAR fetch layer does not need the structured
snapshot — but 4 needs both. 5 touches none of the others and can land at any
point.

The ordering is not arbitrary. Stage 1 is the prerequisite for stage 2 being
worth much: reaching a primary source is of limited use while the figures derived
from it are still discarded at the moment they are formatted. And stage 1 pays
for itself alone, by putting the four metrics with the worst track record under
the same adjudication the market figures already get.
