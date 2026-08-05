# Backtesting: does the pipeline actually add anything?

The README is careful to call TradingAgents "a research scaffold for studying
multi-agent analysis, not a strategy with a fixed, replicable return." This
harness is how you check that on your own configuration, model, and universe.

It answers one question: **do the pipeline's ratings beat baselines that cost
nothing to produce?** Not "did it make money" — over a rising window, anything
net-long makes money.

```bash
# Free. No API key, no LLM calls. Start here.
python -m tradingagents.backtest --start 2024-01-01 --baselines-only

# See what a real run would cost before spending anything.
python -m tradingagents.backtest --start 2025-06-01 --universe mixed --dry-run

# The real thing: resumable, out-of-sample only, full report.
python -m tradingagents.backtest \
    --start 2025-06-01 --universe mixed \
    --knowledge-cutoff 2025-03-01 \
    --cache runs/bt.jsonl --out runs/report.md
```

## How a decision is scored

1. A **grid** of (ticker, date) points is built from a universe and a rolling
   schedule. Every strategy sees the identical grid, which is what makes the
   comparison paired.
2. Each strategy returns one of the five ratings, which maps to a position
   weight (`Buy` +1.0 … `Sell` −1.0; use `--position-map long_only` to floor the
   bearish half at zero).
3. The position is entered at the **next** trading day's close (`--entry-offset 1`,
   the default) and held for `--holding-days` trading days.
4. The contribution is `position × (return − benchmark return)` over that window.

Entry defaults to the next close rather than the decision date's close because an
analysis run against data through the decision date cannot also be executed at
that same close. Setting `--entry-offset 0` reproduces the convention the live
memory log uses; it will flatter results slightly.

## The baselines

| Baseline | What it is | What losing to it means |
|---|---|---|
| `always_buy` | Buy-and-hold: `Buy` on every point | The pipeline does not beat doing nothing |
| `random_uniform` | Coin-flip ratings, seeded | The pipeline carries no information at all |
| `random_matched` | The pipeline's **own** rating distribution, randomly reassigned | The pipeline's only contribution was its average bullishness |

`random_matched` is the one that matters. It is built after the run from the
ratings the pipeline actually produced, so it has exactly the same bullish tilt —
it just assigns those ratings to the wrong points. A pipeline that beats
`always_buy` only because it happened to be less-than-fully-long during a
drawdown will tie this baseline. Beating it requires the ratings to be
*correctly assigned*, which is the only thing worth paying for.

## Reading the output

The report leads with the paired comparison table and a plain-language verdict:

```
| Baseline       | n   | Mean alpha difference | 95% CI              | p     | Beats baseline? |
|----------------|-----|-----------------------|---------------------|-------|-----------------|
| always_buy     | 408 | -0.189%               | [-0.761%, +0.380%]  | 0.513 | no              |
| random_uniform | 408 | +0.189%               | [-0.304%, +0.659%]  | 0.436 | no              |
```

Other columns worth knowing:

- **Mean alpha** is the headline. Mean raw return mostly measures whether the
  market went up.
- **IC** (information coefficient) is the rank correlation between position and
  realized alpha. A constant strategy scores 0 by construction. Note that a
  strategy emitting only `Buy`/`Sell` caps out near 0.87 even when perfectly
  right — the full five-tier spread is needed to reach 1.0.
- **Rating distribution** is the fastest way to spot a degenerate pipeline. If
  it emits `Buy` on 90% of points, it is buy-and-hold in costume whatever its
  headline return says.
- **t** is reported for familiarity but assumes independent observations, which
  the grid violates. Trust the bootstrap interval.

## Statistical choices that keep results honest

**Confidence intervals cluster on the decision date.** Twelve tickers rated on
the same day are not twelve independent observations — they share a market factor
and tend to be right or wrong together. Resampling individual decisions would
narrow the interval by roughly √12 and manufacture significance. The bootstrap
resamples whole dates, with all their tickers.

**Keep `--step-days` at or above `--holding-days`.** Otherwise holding windows
overlap, consecutive observations correlate, and the intervals get optimistic
again. The CLI warns when you do this.

**Unresolvable points are dropped before any strategy runs**, so every strategy
is scored on exactly the same set. A delisted ticker or a too-recent date removes
that point for everyone, not just for the strategy that happened to fail on it.

**Decision dates are clamped to what has settled.** A decision whose holding
window has not closed has no realized return; including it would silently drop
the most recent — and often most volatile — points.

## The knowledge-cutoff problem

This is the one that invalidates most LLM backtests, and the harness cannot fix
it for you — it can only make it visible.

An LLM's weights encode what happened after any date before its training cutoff.
Ask a model to analyze NVDA on 2024-05-10 and it *knows* the next earnings print
was a blowout. Every look-ahead guard in the data layer — and this repo has good
ones, filtering news by publish time in UTC — is powerless against that, because
the leak is in the model, not the data.

So: pass `--knowledge-cutoff` with your model's training cutoff. Points at or
before it are flagged and reported in a separate section rather than dropped,
because **the gap between the contaminated and out-of-sample subsets is itself
the finding.** A pipeline that looks brilliant before the cutoff and average
after it was recalling, not forecasting.

Without `--knowledge-cutoff`, the report says so explicitly and nothing in it
should be read as out-of-sample evidence.

## Reproducible and offline runs

`--prices-dir DIR` loads `<SYMBOL>.csv` (a date column plus `Close`) instead of
calling yfinance. Use it to pin results to a fixed data snapshot, to run against
a survivorship-free vendor extract, or to run in CI with no network access.

The built-in universes are checked in rather than pulled from a live index, since
applying today's index membership to a 2022 window is survivorship bias. They
still carry the milder version of it — these are names that are liquid *today* —
which lifts absolute returns. Paired differences against baselines on the
identical grid are unaffected, which is another reason to read the comparison
table rather than the absolute numbers.

## Ablations

The graph has twelve decision nodes but only four fetch new information; the
other eight re-process the same analyst reports. Whether those eight change the
outcome is an empirical question, and the ablation runner answers it by running
configurations that differ in exactly one respect over an identical grid.

```bash
# Always start here — an ablation is the most expensive thing in this package.
python -m tradingagents.backtest.ablation_cli \
    --start 2025-06-01 --universe smoke --dry-run

# What does each analyst contribute?
python -m tradingagents.backtest.ablation_cli \
    --start 2025-06-01 --suite analysts_drop_one \
    --cache runs/ablation.jsonl --out runs/ablation.md

# A specific pair, cheapest possible comparison
python -m tradingagents.backtest.ablation_cli --start 2025-06-01 --arms all,market
```

### Suites

| Suite | Arms | Question it answers |
|---|---|---|
| `analysts_drop_one` | Full, plus one arm per analyst removed | What does each analyst contribute? |
| `analysts_solo` | Full, plus one arm per analyst alone | How much does one analyst reproduce on its own? |
| `debate_depth` | Bull/Bear rounds at 1, 2, 3 | Is deeper debate worth the tokens? |
| `risk_depth` | Risk-analyst rounds at 1, 2, 3 | Same, for the risk stage |

Every arm is compared against the **reference arm** (the first one, which the
presets make the full pipeline) using the same paired, date-clustered bootstrap
the baseline comparison uses.

One cache file is safe across all arms: arm names are derived from their
configuration (`ta[analysts=market+news]`), so two differently-configured arms
can never collide, and two spellings of the same configuration share cached
decisions instead of paying twice.

### What can and cannot be ablated from config

**Analysts genuinely come out of the graph.** `selected_analysts` removes their
nodes and their tools, so the analyst suites measure real structural changes.

**Debate and risk arms only change depth.** The graph hard-codes at least one
Bull/Bear exchange and one pass through the three risk analysts, so
`debate_depth` measures the marginal value of *more* debate, not the value of
debate versus none. Removing those stages entirely needs a graph change, not a
config flag. The report says so inline, because it is the easiest wrong
conclusion to draw from the table.

### Reading a null result

An arm that removes work and shows no measurable difference means that work was
not paying for itself **on this grid** — that is the actionable direction, and
the asymmetry matters: for a component that costs tokens, "no detectable effect"
argues for dropping it, not keeping it.

But absence of a difference is not proof of equivalence, and at small sample
sizes the two are indistinguishable in the table. The report states the
**resolvable effect** — the confidence interval's half-width — directly under
the comparison:

```
**Resolvable effect**: roughly 0.443% per decision (34 decision dates). A true
difference smaller than that will read as "no measurable difference" here
regardless of whether it is real.
```

Check that number against the effect size you would care about before concluding
anything. In practice, a 12-ticker grid over ~34 decision dates resolves only
fairly large per-decision effects; distinguishing subtler contributions needs a
longer date range, a wider universe, or both — which costs proportionally more
LLM runs.

### Other things worth ablating

Beyond the presets, `AblationArm` takes arbitrary config overrides:

```python
from tradingagents.backtest import AblationArm, reference_arm, run_ablation

arms = [
    reference_arm(),
    AblationArm("cheap managers", config_overrides={"deep_think_llm": "gpt-5-mini"}),
    AblationArm("zero temperature", config_overrides={"temperature": 0.0}),
]
```

## Cost

One evaluation point is one full multi-agent run. A 12-ticker × 26-date grid is
312 runs. Always `--dry-run` first, and always pass `--cache` so an interrupted
run resumes instead of re-spending: every rating is appended to the JSONL cache
the moment it is produced, and a corrupt trailing line from a hard kill is
skipped on reload rather than discarding the file.

## Programmatic use

```python
from tradingagents.backtest import (
    AlwaysRating, Backtest, TradingAgentsStrategy, build_backtest_config, build_grid,
)

grid = build_grid(
    ["AAPL", "MSFT"], start="2025-06-01", holding_days=21, step_days=21,
    knowledge_cutoff="2025-03-01",
)
result = Backtest(grid, cache_path="runs/bt.jsonl").run(
    TradingAgentsStrategy(config=build_backtest_config()),
    [AlwaysRating("Buy")],
)

print(result.metrics["trading_agents"].mean_alpha)
for comparison in result.comparisons:
    print(comparison.baseline, comparison.mean_difference, comparison.significant)
```

`build_backtest_config()` disables the memory log and checkpointing. The memory
log matters: with it enabled, decisions later in the grid are informed by
realized outcomes of earlier ones — information a live run genuinely has, but
which makes later points non-comparable to earlier ones.

## A note on what a null result means

If the pipeline ties its baselines, that is a real finding, not a failed
experiment — particularly given what a run costs. The harness is built not to
hide it: it never drops a strategy for underperforming, states the verdict in
words, and ships a positive-control test (an oracle strategy that peeks at the
outcome) proving the comparison is capable of detecting an edge when one exists.
