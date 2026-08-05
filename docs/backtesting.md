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

`--analysts` restricts the pipeline to a subset, which prices what the rest
contribute:

```bash
# All four analysts (default)
python -m tradingagents.backtest --start 2025-06-01 --cache runs/full.jsonl --out runs/full.md

# Market analyst only
python -m tradingagents.backtest --start 2025-06-01 --analysts market \
    --cache runs/market.jsonl --out runs/market.md
```

Use separate cache files: the cache is keyed on strategy name and point, not on
configuration, so pointing two different configurations at one cache will serve
the first one's decisions to the second.

Worth ablating, in rough order of how much they cost to run:

- The four analysts individually (`--analysts`).
- Debate depth — set `max_debate_rounds` / `max_risk_discuss_rounds` in the
  config. Only four of the graph's twelve nodes fetch new data; the rest
  re-process the same analyst reports, so it is worth measuring whether the
  extra rounds move the result at all.
- Deep vs quick model for the two manager nodes.

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
