# HTML reports

Two things render to the browser: a finished **analysis run**, and **backtest /
ablation results** with charts.

Every page is a **single self-contained file** — inline CSS, inline SVG, one
small theme-toggle script, no external requests of any kind. It opens from
`file://`, survives being emailed as an attachment, prints sensibly, and can be
archived next to the run it describes. There is no server and nothing to install
beyond the package itself.

## Automatic generation

Saving a run from the CLI writes `complete_report.html` beside `complete_report.md`,
and prints a `file://` link you can paste straight into a browser:

```
✓ Report saved to: /home/you/reports/NVDA_20260805_120000
  Complete report: complete_report.md
  Browser version: complete_report.html
  Open: file:///home/you/reports/NVDA_20260805_120000/complete_report.html
```

`TradingAgentsGraph.save_reports()` produces it too — both paths go through the
same writer, so a headless run gets the same artifacts a CLI run does.

Markdown remains the source of truth. The HTML is generated *from* it, and if
generation fails the markdown tree is already on disk and is kept: the failure
is logged, `ReportPaths.html` comes back `None`, and the CLI simply does not
print a browser line. A rendering bug must never turn a completed analysis —
which cost real money — into a lost report.

To turn it off:

```bash
tradingagents analyze --no-html          # one run
export TRADINGAGENTS_REPORT_HTML=false   # always
```

```python
config["report_html"] = False            # programmatically
write_report_bundle(state, "NVDA", path, html=False)   # per call
```

Use `write_report_bundle` when you want both paths back; `write_report_tree`
still returns just the markdown path, unchanged, for existing callers.

## Reading an analysis run in the browser

```bash
# a report directory written by the CLI or save_reports()
python -m tradingagents.webreport ~/.tradingagents/logs/reports/NVDA_20260805_120000

# an older run that only kept the state log
python -m tradingagents.webreport \
    ~/.tradingagents/logs/NVDA/TradingAgentsStrategy_logs/full_states_log_2026-08-05.json

# any markdown file
python -m tradingagents.webreport report.md -o report.html
```

You get section navigation, styled tables (every analyst prompt asks for a
summary table, and those are what read worst as plain text), and both light and
dark themes with a toggle.

Markdown stays the source of truth — the HTML is generated *from* it and never
replaces it, so a page can always be regenerated from an archived run.

Report bodies are LLM output, so they are rendered as untrusted content: raw
HTML in a report is escaped rather than executed.

## Charted backtest results

```bash
python -m tradingagents.backtest --start 2025-06-01 --universe mixed \
    --cache runs/bt.jsonl --out runs/report.md --html runs/report.html

python -m tradingagents.backtest.ablation_cli --start 2025-06-01 \
    --suite analysts_drop_one --cache runs/abl.jsonl --html runs/ablation.html
```

`--html` is additive: pass it alongside `--out` and you get both formats from one
run.

### The three charts

**Confidence intervals** lead the page, above the metrics table. The claim this
whole package exists to support is "the difference from the baseline excludes
zero", and that is a statement about a picture — whether a bar crosses a line.
Putting the per-strategy return table first invites the opposite reading:
comparing absolute alphas across rows, which is exactly the mistake the paired
design prevents.

**Running mean alpha** answers what a single mean cannot: was the edge steady, or
one lucky window? A line that jumps once and then flattens made its whole result
in two weeks. It is a running *mean* rather than a running sum, so strategies
scored on different decision counts stay comparable.

**Rating mix** is a diverging bar — bullish right, bearish left, Hold straddling
the centre. The five-tier scale is ordered and signed, so it takes a diverging
ramp rather than categorical hues. A pipeline that emits Buy almost everywhere
shows up instantly as one solid block, which is the fastest way to spot
buy-and-hold wearing a costume.

## Colour

The palette is not chosen by eye. It comes from the project's
data-visualisation reference and was checked with its validator:

| Role | Light | Dark |
|---|---|---|
| Strategy series (fixed order, never cycled) | `#2a78d6` `#eb6834` `#1baf7a` `#eda100` | `#3987e5` `#d95926` `#199e70` `#c98500` |
| Rating scale, Buy → Sell | `#184f95` `#5598e7` `#898781` `#e34948` `#8f2020` | `#b7d3f6` `#3987e5` `#504f4a` `#e66767` `#f7b3b3` |

The categorical series pass every check in both modes (worst adjacent
colour-vision-deficiency ΔE 9.1 light / 8.4 dark; normal-vision 22.9 / 19.8).

The rating ramp is diverging, so it is validated as such: each arm passes as a
one-hue ordinal ramp, and across the scale the checks that govern readability
pass in both modes (CVD ΔE 8.7 light / 15.9 dark; normal-vision 15.3 / 17.9).
Two categorical-only checks — lightness band and chroma floor — do fail on it,
and that is expected rather than ignored: a diverging ramp is *required* to have
a low-chroma neutral midpoint and outer steps beyond the categorical band. Those
two properties are what make it diverging.

The neutral midpoint sits below 3:1 against the surface in both themes, which
obligates relief rather than being dismissable. Both forms are shipped: visible
percentage labels on every segment wide enough to hold one, and the equivalent
data table beneath every chart.

Series colour is never the only channel. Every chart with two or more series has
a legend, four or fewer are also directly labelled, significance is stated in
words in the hover text and the table, and the rating scale's meaning is carried
by position around the centre line as much as by hue.

## Programmatic use

```python
from pathlib import Path
from tradingagents.webreport import (
    markdown_to_page, render_ablation_html, render_backtest_html, render_report_dir,
)

Path("report.html").write_text(render_backtest_html(result), encoding="utf-8")
Path("ablation.html").write_text(render_ablation_html(ablation_result), encoding="utf-8")
Path("run.html").write_text(render_report_dir("~/.tradingagents/logs/reports/NVDA_x"), encoding="utf-8")
Path("any.html").write_text(markdown_to_page(some_markdown, title="Title"), encoding="utf-8")
```

## What this is not

There is no web server, no live dashboard, and no way to launch a run from the
browser. These are static artifacts of runs that already happened. Triggering
analyses remotely and streaming node-by-node progress would need a real backend
(the pipeline takes minutes per run), plus concurrency, state, and API-key
handling — a much larger piece of work, and a different one.

The interactive interface remains the Rich terminal UI (`tradingagents`).
