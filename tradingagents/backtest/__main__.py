"""CLI: ``python -m tradingagents.backtest``.

Two modes worth knowing about:

``--baselines-only``
    Runs no LLM at all. Costs nothing, needs no API key, and establishes what
    the null distribution looks like on your grid before you spend money.

``--dry-run``
    Prints the grid and the estimated LLM call count, then exits. Worth running
    first: the full pipeline is one multi-agent run per point, so a 12-ticker,
    24-date grid is 288 runs.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .grid import build_grid, latest_settled_date
from .harness import Backtest
from .prices import PriceCache
from .report import render_report, render_summary
from .strategies import (
    AlwaysRating,
    TradingAgentsStrategy,
    UniformRandomRating,
    build_backtest_config,
)
from .universe import DEFAULT_UNIVERSE, UNIVERSES, resolve_universe


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tradingagents.backtest",
        description="Score the TradingAgents pipeline against random and buy-and-hold baselines.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  # free: what do the baselines alone do on this grid?\n"
            "  python -m tradingagents.backtest --start 2024-01-01 --baselines-only\n\n"
            "  # the real thing, resumable, out-of-sample only\n"
            "  python -m tradingagents.backtest --start 2025-06-01 \\\n"
            "      --universe mixed --knowledge-cutoff 2025-03-01 \\\n"
            "      --cache runs/bt.jsonl --out runs/report.md\n"
        ),
    )
    parser.add_argument("--start", required=True, help="first decision date, YYYY-MM-DD")
    parser.add_argument("--end", help="last decision date; clamped to the last settled date")
    parser.add_argument(
        "--universe",
        default=DEFAULT_UNIVERSE,
        help=f"named universe ({', '.join(sorted(UNIVERSES))}) or a comma-separated ticker list",
    )
    parser.add_argument("--holding-days", type=int, default=21, help="trading days held (default 21)")
    parser.add_argument(
        "--step-days", type=int, default=21,
        help="calendar days between decision dates (default 21; keep >= holding days to avoid "
             "overlapping windows)",
    )
    parser.add_argument("--benchmark", default="SPY", help="alpha benchmark (default SPY)")
    parser.add_argument(
        "--position-map", default="signed", choices=("signed", "long_only"),
        help="signed treats Sell as a short; long_only floors negatives at zero",
    )
    parser.add_argument(
        "--entry-offset", type=int, default=1,
        help="trading days between the decision date and entry (default 1: the next close, "
             "since an analysis run through the decision date cannot execute at that same close)",
    )
    parser.add_argument(
        "--knowledge-cutoff",
        help="model training cutoff YYYY-MM-DD; points at or before it are flagged and "
             "reported separately as not out-of-sample",
    )
    parser.add_argument(
        "--prices-dir",
        help="load prices from <SYMBOL>.csv files here instead of yfinance (needs a date "
             "column and a Close column). Use for reproducible or survivorship-free data, "
             "or to run without network access",
    )
    parser.add_argument("--cache", help="JSONL decision cache; enables resuming an interrupted run")
    parser.add_argument("--out", help="write the full markdown report here")
    parser.add_argument("--seed", type=int, default=0, help="baseline/bootstrap seed (default 0)")
    parser.add_argument(
        "--bootstrap-iterations", type=int, default=10_000,
        help="bootstrap resamples for the confidence intervals (default 10000)",
    )
    parser.add_argument(
        "--baselines-only", action="store_true",
        help="skip the LLM pipeline entirely; runs the baselines against each other",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the grid and estimated cost, then exit"
    )
    parser.add_argument(
        "--analysts",
        help="comma-separated analyst subset for the pipeline (default: all four). Use this to "
             "ablate — e.g. --analysts market to price the other three analysts' contribution",
    )
    parser.add_argument("--verbose", action="store_true", help="debug logging")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    tickers = resolve_universe(args.universe)
    try:
        grid = build_grid(
            tickers,
            start=args.start,
            end=args.end,
            holding_days=args.holding_days,
            step_days=args.step_days,
            knowledge_cutoff=args.knowledge_cutoff,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print(
            f"hint: with a {args.holding_days}-day hold, the latest usable decision date is "
            f"{latest_settled_date(args.holding_days)}",
            file=sys.stderr,
        )
        return 2

    if args.step_days < args.holding_days:
        print(
            f"warning: step-days ({args.step_days}) < holding-days ({args.holding_days}); "
            "holding windows overlap, so observations are correlated and the confidence "
            "intervals will be optimistic.",
            file=sys.stderr,
        )

    if args.dry_run:
        return _dry_run(grid, args)

    baselines = [
        AlwaysRating("Buy"),
        UniformRandomRating(seed=args.seed),
    ]

    if args.baselines_only:
        # With no pipeline to compare against, promote a baseline to the
        # "strategy" slot so the harness still produces a paired table. This is
        # the null experiment: whatever margin shows up here is noise, and it
        # calibrates how large a real effect would have to be to stand out.
        strategy = UniformRandomRating(seed=args.seed + 1, name="random_uniform_b")
        baselines = [AlwaysRating("Buy"), UniformRandomRating(seed=args.seed)]
        include_matched = False
    else:
        selected = (
            tuple(a.strip() for a in args.analysts.split(",") if a.strip())
            if args.analysts
            else None
        )
        strategy = TradingAgentsStrategy(
            config=build_backtest_config(), selected_analysts=selected
        )
        include_matched = True

    price_cache = None
    if args.prices_dir:
        price_cache = PriceCache()
        price_cache.load_csv_dir(args.prices_dir, [*grid.tickers, args.benchmark])
        if price_cache.failures():
            for symbol, reason in price_cache.failures().items():
                print(f"warning: {symbol}: {reason}", file=sys.stderr)

    backtest = Backtest(
        grid,
        price_cache=price_cache,
        benchmark=args.benchmark,
        position_map=args.position_map,
        entry_offset=args.entry_offset,
        cache_path=args.cache,
        bootstrap_iterations=args.bootstrap_iterations,
        seed=args.seed,
    )

    try:
        result = backtest.run(
            strategy,
            baselines,
            include_matched_random=include_matched,
            progress=_progress if args.verbose else None,
        )
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(render_summary(result))

    if args.out:
        out_path = Path(args.out).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(render_report(result), encoding="utf-8")
        print(f"\nFull report written to {out_path}")

    return 0


def _dry_run(grid, args) -> int:
    print(f"Grid: {len(grid)} points ({len(grid.tickers)} tickers x {len(grid.dates)} dates)")
    print(f"  tickers: {', '.join(grid.tickers)}")
    print(f"  dates:   {grid.start} .. {grid.end} (every {args.step_days} calendar days)")
    print(f"  holding: {grid.holding_days} trading days, entry +{args.entry_offset}")
    if grid.knowledge_cutoff:
        print(
            f"  cutoff:  {grid.knowledge_cutoff} -> {grid.contaminated_count} contaminated, "
            f"{len(grid.clean())} out-of-sample"
        )
        if not grid.clean():
            print("  WARNING: every point precedes the knowledge cutoff; nothing is out-of-sample.")
    if args.baselines_only:
        print("\nBaselines only: 0 LLM runs.")
    else:
        print(f"\nEstimated cost: {len(grid)} full multi-agent runs (baselines are free).")
        print("Pass --cache to make the run resumable before starting.")
    return 0


def _progress(strategy_name: str, point, rating: str) -> None:
    print(f"  [{strategy_name}] {point.key} -> {rating}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
