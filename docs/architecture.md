# Architecture: what runs, in what order, and what checks it

One `propagate(ticker, date)` call walks a LangGraph `StateGraph` containing
twelve decision nodes, and returns a report plus a five-tier rating.

This document describes that walk. It also describes something the node diagram
does not show: **three gates that contain no LLM at all**, and why the pipeline
needs them in places the agents cannot cover themselves.

## The pipeline

```text
propagate(ticker, date)
   │
   ▼
dataflows/ ─ yfinance · Alpha Vantage · FRED · StockTwits · Reddit · Polymarket
   │
   ├── GATE 1 ─ validators: reconcile a vendor figure against that vendor's own
   │            line items, filter look-ahead, label every number's unit,
   │            period, and settlement state
   ▼
1  ANALYSTS (sequential)   Market → Sentiment [G2] → News → Fundamentals
   │                       each is an agent ⇄ ToolNode loop, then Msg Clear
   │                       writes: market_report, sentiment_report, …
   ▼
2  RESEARCH DEBATE         Bull ⇄ Bear, until count ≥ 2 × max_debate_rounds
   │                       writes: investment_debate_state
   ▼
3  RESEARCH MANAGER [G2]   deep model, adjudicates
   │                       writes: investment_plan
   ▼
4  TRADER [G2]             writes: trader_investment_plan
   ▼
5  RISK DEBATE             Aggressive → Conservative → Neutral,
   │                       until count ≥ 3 × max_risk_discuss_rounds
   │                       writes: risk_debate_state
   ▼
6  PORTFOLIO MANAGER [G2]  deep model
   │                       writes: final_trade_decision
   ▼
   report assembly ─────── GATE 3 ─ report_lint + code-revision stamp
   │
   ▼
   Markdown / HTML

[G2] = GATE 2. The node's answer must satisfy a pydantic schema. A rejection
       earns one retry with the complaint fed back; anything that still falls
       through is labelled unvalidated in the output.
```

The wiring lives in `graph/setup.py`; every routing decision lives in
`graph/conditional_logic.py`. Both are short and worth reading before changing
anything.

| Stage | Nodes | Model tier | Loop control |
|---|---|---|---|
| 1 | Market / Sentiment / News / Fundamentals | quick | until no more tool calls |
| 2 | Bull ⇄ Bear | quick | `max_debate_rounds` |
| 3 | Research Manager | **deep** | single pass |
| 4 | Trader | quick | one schema retry |
| 5 | Aggressive → Conservative → Neutral | quick | `max_risk_discuss_rounds` |
| 6 | Portfolio Manager | **deep** | single pass |

The two-tier model split is a cost decision. The debate stages burn the most
tokens and do the least irreversible work, so they run on
`quick_thinking_llm`; only the two nodes that actually adjudicate use
`deep_thinking_llm`. Both are configurable through `TRADINGAGENTS_*` environment
variables without touching code.

### Every shared router carries a complete path_map

Routers that several edges share — the debate router, the risk router — are
registered with a path map naming *every* target they can return, not just the
ones expected at that edge:

```python
DEBATE_PATH_MAP = {
    "Bull Researcher": "Bull Researcher",
    "Bear Researcher": "Bear Researcher",
    "Research Manager": "Research Manager",
}
```

These routers read speaker labels out of state. Prompt edits, translation, or a
refactor can drift those labels, and a fall-through return with an incomplete
path map crashes LangGraph mid-run. Mapping everything turns a would-be crash
into a wrong-but-recoverable hop (#1088).

## State is a dict, not a conversation

Nodes do not pass message history to each other. They read and write a
structured `AgentState` (`agents/utils/agent_states.py`):

```python
{
    "company_of_interest": "NVDA",
    "trade_date": "2026-08-07",
    "instrument_context": ...,      # resolved once at run start
    "past_context": ...,            # reflections recalled from the memory log
    "verified_market_block": ...,   # snapshots for the agents after the
    "verified_fundamentals_block": ...,  # analysts; also resolved once

    "market_report": ...,           # stage 1 writes four of these
    "sentiment_report": ...,
    "news_report": ...,
    "fundamentals_report": ...,

    "investment_debate_state": {    # stage 2
        "bull_history": ..., "bear_history": ...,
        "current_response": ..., "count": 0,
    },
    "risk_debate_state": {          # stage 5, plus latest_speaker
        "latest_speaker": ..., "count": 0, ...
    },
}
```

Routing reads two fields — `count` and `latest_speaker`. Whether to debate
another round is therefore a pure function of state, independent of what the
model said, which is also what makes the graph checkpointable.

### A checkpoint's thread ID must fold in the graph's shape

Keying a resume on `ticker + date` alone is not enough. Run four analysts, crash
halfway, then rerun with two analysts, and the old checkpoint resumes happily —
on a graph that no longer exists.

```python
# trading_graph.py — everything that changes the graph's shape
def _run_signature(self, asset_type):
    return "|".join([
        "analysts=" + ",".join(self.selected_analysts),
        f"debate={self.config['max_debate_rounds']}",
        f"risk={self.config['max_risk_discuss_rounds']}",
        f"asset={asset_type}",
    ])

# checkpointer.py — a different signature is a different thread
def thread_id(ticker, date, signature=""):
    base = f"{ticker.upper()}:{date}"
    if signature:
        base = f"{base}:{signature}"
    return hashlib.sha256(base.encode()).hexdigest()[:16]
```

The general form: any cache premised on "the same inputs" needs a key covering
everything that changes the computation, not just the data identifiers.
Configuration rarely feels like input, which is why this is easy to miss
(#1089).

## Analyst nodes and the clear step

Analysts run sequentially. If each one's tool traffic stayed in context, the
fourth would inherit three analysts' worth of tool calls and results.

```text
   Market Analyst ──── tool_calls ≠ ∅ ───▶ ToolNode
        ▲                                     │
        └────────── results appended ─────────┘
        │
        │ no tool_calls
        ▼
   Msg Clear ──── report survives, tool traffic deleted ───▶ next analyst
```

`should_continue_market()` and its siblings check one thing: whether the last
message carries tool calls. When it does not, the analyst is done and the run
passes through a `create_msg_delete()` node before the next analyst starts. What
crosses that boundary is the finished report, nothing else.

## What the debate could not catch, and what changed

The bull and bear researchers each read the same four fields:

```python
market_research_report = state["market_report"]
sentiment_report       = state["sentiment_report"]
news_report            = state["news_report"]
fundamentals_report    = state["fundamentals_report"]
```

For a long time that was *all* either of them read, and neither had a route back
to the source data. A wrong figure reaching those reports became a **shared
premise**: the bull argued from it, the bear argued against the conclusion rather
than the number, and the adversarial structure never touched it. Adversarial
agents can contest interpretations of a fact; they cannot contest the fact itself
when both sides were handed it and neither can look it up.

This was not hypothetical. Every numeric defect found in shipped reports
survived a debate that ran exactly as designed:

| Defect | What the debate did |
|---|---|
| Stop-loss placed *above* a long's entry price | Quoted it; arithmetic was correct, direction was not |
| Gross margin "up 2.3 percentage points" (actually 0.98) | Cited it across three rounds |
| Simplified FCF quoted as proof the fabs self-fund | Neither side noted the company's own figure has the opposite sign |
| Segment ASP restated as company-wide pricing power | Treated as established |
| A StockTwits number restated until it read as fact | Made it a pillar of the rating |

The first conclusion is structural and still holds: **verification has to be its
own layer, and it cannot be delegated to the agents.** That is what the three
gates below are.

The second conclusion is that the debate was being asked to catch these without
the means to. `agents/utils/verified_evidence.py` resolves both snapshots once at
run start, stores them on the state, and hands them to every agent after the
analysts:

| Agent | Verified market levels | Verified fundamentals |
|---|---|---|
| Bull / Bear Researcher | yes | yes |
| Research Manager | yes | yes |
| Trader | its own execution-flavoured block | yes |
| Portfolio Manager | its own execution-flavoured block | yes |

The rule attached to the block is what makes it more than extra context: where a
report, a debate turn, or a social post disagrees with a verified figure, **the
verified figure stands and the conflict gets named** — not averaged, not silently
resolved. A debater can now argue *about* a premise rather than only from it,
which is the one place in the pipeline where a wrong premise can still be
challenged; every check after this point runs on prose that already contains it.

Two boundaries worth knowing. The Trader and the Portfolio Manager already render
their own price-levels block carrying the execution rule about ATR distances, so
they receive `include_market=False` and are not handed the same numbers twice
under two headings. And the three risk debators are **not** on this path yet:
they argue about a proposal the Trader has already priced against verified
levels, so the same laundering is possible but one step further from the source.
Extending it there is a follow-up, not an oversight.

A state built without the blocks — a bare programmatic state, a test — degrades
to the explicit "unavailable" notices rather than resolving them mid-graph, the
same rule `get_instrument_context_from_state` follows for the instrument context.
The degradation is visible in the prompt, so an agent told the figures are
unavailable declines to state them instead of quietly inventing them.

## The three gates

| Gate | Where | Catches | On failure |
|---|---|---|---|
| 1 | Before analysts see a number | Wrong period, unit, or definition; totals that do not reconcile | Prints operands alongside the figure, or refuses to state it |
| 2 | Sentiment Analyst, Research Manager, Trader, Portfolio Manager output | Internal incoherence a schema can express | One retry with the complaint fed back, then a visible unvalidated notice |
| 3 | Assembled report | One number under two incompatible labels | Appends a warning block; stamps the code revision |

None of the three asks an LLM anything.

### Gate 1 — data validators

`dataflows/fundamentals_validator.py` recomputes vendor figures from the
vendor's own line items, resolves each ratio's period rather than assuming one,
and prints derived values with their operands: `+0.98 pp (40.36% − 39.38%)`
rather than a bare delta the reader has to trust.

Two habits from this module generalise:

**Name the definition, not just the metric.** Every free-cash-flow column reads
`simplified FCF (OCF − capex)`, with a note that a company-defined measure
exists, comes from the filings, is unavailable from this vendor, and can carry
the opposite sign. The gap is disclosed rather than quietly resolved in
whichever direction happens to suit.

**Ask the exchange's clock, not the host's.** `dataflows/session_status.py`
classifies the newest daily bar as `FINAL`, `IN-PROGRESS`, or `UNKNOWN` from the
instrument's own exchange calendar. Yahoo publishes a partial candle while a
session is open — its close is the last trade, not the settlement price — and
nothing downstream could otherwise tell that row apart from a settled one.

```python
SETTLEMENT_BUFFER_MINUTES = 15
```

The closing auction and late prints can still move the official close for a few
minutes after the bell. Erring long means a settled bar is briefly labelled
`IN-PROGRESS`, which is the harmless direction of the error. Choosing which way
to be wrong, deliberately, recurs throughout this codebase.

### Gate 2 — structured output

`agents/utils/structured.py` is the smallest of the three and the most
generally useful. Four agents share it:

| Agent | Schema |
|---|---|
| Sentiment Analyst | `SentimentReport` |
| Research Manager | `ResearchPlan` |
| Trader | `TraderProposal` |
| Portfolio Manager | `PortfolioDecision` |

The pattern it replaced looked reasonable:

```python
try:
    result = structured_llm.invoke(prompt)
    return render(result)
except Exception:
    return plain_llm.invoke(prompt).content   # the bug
```

That `except` conflates two unrelated failures:

- **The provider cannot do structured output at all.** A capability problem.
  Retrying is pointless.
- **The schema rejected the model's answer.** The model got something wrong,
  and the rejection message describes exactly what.

Treating the second as the first publishes a proposal the validator just refused
— in a section that reads exactly like one where every check passed. The current
flow separates them:

```text
1  structured_llm.invoke(prompt)
        │ success ─────────────────────────▶ render(result)  ✓
        │ failure
        ▼
2  _is_validation_failure(exc)?
        │ no  ── capability or transport ──▶ no retry
        │ yes ── the schema rejected the content
        ▼
3  retry once, with the pydantic error appended:
   "This is a real inconsistency in the proposal, not a formatting problem.
    Fix the underlying values — do not restate the same numbers."
        │ success ─────────────────────────▶ render(retried)  ✓
        │ failure
        ▼
4  plain_llm.invoke() + _mark_unvalidated(...)
```

Whatever reaches step 4 is prefixed with a notice naming the checks that did not
run — for the trader, position-intent coherence, stop-loss direction, and ATR
distance. The reason string is collapsed to a single line first, because a
multi-line pydantic error would break out of the markdown blockquote and bury
the warning in stack-trace furniture.

Two smaller notes on this path:

**Say the tool constraint out loud.** Schema-only structured output binds
exactly one tool, so a model that reaches for a search tool emits an unknown
tool call and the whole attempt is discarded. Agents on this path state
`NO_EXTERNAL_TOOLS` in the prompt rather than relying on the binding alone
(#1130).

**Commentary is not output.** The unvalidated notice is *prepended*, so a schema
error quoting the allowed values (`Input should be 'Buy', 'Overweight', …`) sits
above the decision it describes — where a line-scanning parser will read it as
the decision. Both `parse_rating()` and `parse_trader_action()` therefore skip
blockquoted lines. If you annotate model output, pick a syntactic boundary every
downstream parser agrees on.

### Gate 3 — report lint and provenance

`report_lint.py` binds each number in the finished report to the nearest metric
label, groups the readings, and reports a metric carrying two incompatible
values. Most of its code is not the binding — it is the cases that must *not*
bind:

```text
50-day SMA ($99.43 vs $110.60)   a `vs` pair: which side is the metric is not
                                 recoverable from position, so skip both
ATR (14): 8.09                   a bracketed group of bare integers after an
                                 indicator name is a lookback parameter; step
                                 over it and take the reading that follows
ATR ≈8                           ≈ ~ 約 约 about approximately — the writer
                                 rounding, not a second measurement
```

All three follow one rule: **when in doubt, lose a reading rather than invent a
conflict.** A linter that cries wolf on a correct report makes the warning block
the least trustworthy thing in the output, which is the exact failure it exists
to prevent.

Three implementation details that were each a real bug:

```python
# The trailing guard must not be a blanket (?![\d,]). With that form,
# "at $111.52, confirming" fails on the full match, backtracks, and matches
# "111" — which then "conflicts" with the 111.52 stated elsewhere.
_NUMBER_RE = re.compile(rf"(?<![\d,])({_NUMBER})(?!\d|,\d{{3}})")

# Financial prose writes a negative as "-$2.54B", with the currency mark
# between sign and digits. Requiring the mark keeps this away from ordinary
# hyphens like "1-1.5x ATR".
_DETACHED_MINUS_RE = re.compile(r"[-−–]\s*[$¥€£]\s*$")

# Tolerance needs an absolute floor as well as a relative one.
def _within_rounding_tolerance(stated, computed):
    return abs(stated - computed) <= max(0.05, 0.005 * abs(computed))
```

The provenance stamp rides on the report header:

```text
Generated: 2026-08-07 22:24:05 · code fd0dc45
```

It resolves the revision from **the package's own location**, not the working
directory, because those are what diverge: a frozen `pip install .` copy in
site-packages and a current checkout can sit on the same machine, and asking the
shell's cwd would report the fix as present on every run that lacked it. A
virtualenv usually lives inside the project, so `git -C` on an installed copy
walks up and returns the checkout's SHA — recreating the confusion in a form
that now looks authoritative. Tracked-ness separates the two: an untracked
location reports `unknown` and names the path it was imported from. An
uncommitted tree reads `fd0dc45+local-changes`.

## Evidence tiers

Gates catch arithmetic. They do not catch a number that is *correctly copied
from an unreliable source*, and that failure has its own module.

A StockTwits post claimed capex had doubled and free cash flow had fallen 39%.
The sentiment analyst relayed it correctly, as retail opinion — that is its job.
The bear researcher, the research manager, and the portfolio manager then each
restated it, and by the fourth restatement it was a pillar of an Underweight
rating. The same report's own cash-flow table showed free cash flow up 180% year
on year and capex up 53%. No step lied; the attribution simply wore off.

`agents/utils/evidence_policy.py` adds one marking rule for the agent that
touches social sources, and one carrying rule for everyone downstream:

| Tier | Source | How it may be used |
|---|---|---|
| Verified | Verified market or fundamentals snapshot | Cite freely, with the snapshot's period |
| Reported | A named outlet or analyst | Cite with the attribution attached |
| Unverified | Marked `[UNVERIFIED — social post]`, or sourced only from StockTwits/Reddit | Only as a description of what participants believe — never as fact, never as support for a recommendation |

The sentiment analyst also lists every marked claim in an
`unverified_numeric_claims` field so downstream agents can see them without
re-parsing prose.

The load-bearing sentence is the one about repetition:

> a number that arrived unverified stays unverified no matter how many times it
> has been repeated in the debate history.

And where tiers disagree, the verified figure stands and **the conflict is
named** rather than averaged. Reconciling two contradictory numbers into one
plausible sentence is a thing language models do well and should not be doing
here.

### Scope travels with the figure

A related rule, in the same module. Intel's 10-Q reported *server-product* ASP
up 48% year on year, driven mainly by a richer product mix. That reached a
report as "Intel CPU prices up 48%", and from there as evidence of pricing power
across the business — a claim the source does not make. Nothing was misquoted:
the number survived and its scope did not.

`SCOPE_DISCIPLINE` travels with the news analyst, where the boundary is first
dropped, and with every downstream agent. It requires keeping which part of the
business a figure covers, what drove it, and the comparison basis.

## Which rules can be enforced, and which are only asked for

Unlike the arithmetic guards, scope discipline is prompt-level. There is no
source text to diff against at runtime, so it cannot be enforced the way a
recomputed margin can. That distinction is worth making explicitly whenever a
rule is added:

| Rule | Enforceable at runtime | Why |
|---|---|---|
| Gross margin = gross profit / revenue | Yes | Both operands are in hand; recompute |
| Stop-loss direction matches position intent | Yes | A relation between fields; encode in the schema |
| One metric may not carry two values | Yes | The finished text is its own reference |
| Vendor operating income matches its line items | Yes | Recompute from the vendor's own figures |
| A figure's scope may not be widened | **No** | No source text to diff against |
| An unverified number may not become a fact | Partly | The marker is detectable; "used as support" is not |

Before adding a rule, ask whether it can be a pure function. If it can, write the
function — a prompt rule gets diluted, overridden by later instructions, and
drifts when the model changes. If it genuinely cannot, label it as prompt-level
so nobody builds on it expecting determinism.

## Refusing to guess

Four places deliberately return "cannot tell" rather than a plausible answer:

| Site | Ambiguity | Choice |
|---|---|---|
| `parse_trader_action()` | Two conflicting action labels in the prose | `UNDETERMINED` |
| `report_lint` — `vs` pairs | Which side is the metric | Drop both readings |
| `report_lint` — bracket groups | A round number in brackets | Treat as a parameter, lose the reading |
| `provenance` | Import path is untracked | Report `unknown` plus the path |

`parse_trader_action()` explains the shared reasoning: it runs only when the
structured path fell back and the rendered `FINAL TRANSACTION PROPOSAL` line was
lost with it. Putting a guessed direction where a validated one used to be is
worse than leaving a gap, because the two are indistinguishable to a reader.

## Does any of this earn its cost?

The graph has twelve decision nodes, but only four of them fetch new
information; the other eight re-process the same analyst reports. Whether those
eight change the outcome is an empirical question, and
`tradingagents/backtest/ablation.py` is how you answer it on your own models and
universe. See [`docs/backtesting.md`](backtesting.md) for the methodology.

Note one honest limit recorded there: `selected_analysts` genuinely removes
nodes and their tools, so analyst ablations measure real structural changes. The
debate stages only vary in *depth* — the graph hard-codes at least one Bull/Bear
exchange and one pass through the three risk analysts — so `debate_depth`
measures the marginal value of more debate, not of debate versus none.

## Where to look

| Path | What it holds |
|---|---|
| `graph/setup.py` | The wiring. Start here to change the flow |
| `graph/conditional_logic.py` | Every routing decision, as pure functions |
| `graph/checkpointer.py` | Thread IDs and resume, with `_run_signature` |
| `agents/utils/agent_states.py` | The full shape of `AgentState` |
| `dataflows/fundamentals_validator.py` | Gate 1 |
| `dataflows/session_status.py` | Bar settlement, three-state |
| `agents/utils/structured.py` | Gate 2 |
| `agents/schemas.py` | `TraderProposal` and the other pydantic schemas |
| `report_lint.py` | Gate 3 |
| `provenance.py` | The code-revision stamp |
| `agents/utils/evidence_policy.py` | Evidence tiers and scope discipline |
| `agents/utils/verified_evidence.py` | The snapshots handed to agents after the analysts |
| `agents/utils/rating.py` | The five-tier vocabulary and its parsers |
| `backtest/ablation.py` | Pricing what each component contributes |
