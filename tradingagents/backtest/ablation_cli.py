"""CLI: ``python -m tradingagents.backtest.ablation_cli``.

Ablation is the most expensive thing in this package — every arm costs a full
pass over the grid — so the defaults are conservative and ``--dry-run`` reports
the total before anything is spent. One shared cache serves all arms safely
because arm names are derived from their configuration.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .ablation import (
    DEFAULT_SUITE,
    SUITES,
    AblationArm,
    dedupe_arms,
    estimated_runs,
    resolve_suite,
)
from .ablation_report import render_ablation_report, render_ablation_summary
from .grid import build_grid, latest_settled_date
from .harness import Backtest
from .prices import PriceCache
from .universe import DEFAULT_UNIVERSE, UNIVERSES, resolve_universe


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tradingagents.backtest.ablation_cli",
        description="Measure what each part of the TradingAgents pipeline contributes.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  # always start here: how many runs would this cost?\n"
            "  python -m tradingagents.backtest.ablation_cli \\\n"
            "      --start 2025-06-01 --universe smoke --dry-run\n\n"
            "  # what does each analyst contribute?\n"
            "  python -m tradingagents.backtest.ablation_cli \\\n"
            "      --start 2025-06-01 --suite analysts_drop_one \\\n"
            "      --cache runs/ablation.jsonl --out runs/ablation.md\n\n"
            "  # a specific pair, cheapest possible comparison\n"
            "  python -m tradingagents.backtest.ablation_cli \\\n"
            "      --start 2025-06-01 --arms all,market\n"
        ),
    )
    parser.add_argument("--start", required=True, help="first decision date, YYYY-MM-DD")
    parser.add_argument("--end", help="last decision date; clamped to the last settled date")
    parser.add_argument(
        "--suite", default=DEFAULT_SUITE,
        choices=sorted(SUITES),
        help=f"preset arm set (default {DEFAULT_SUITE})",
    )
    parser.add_argument(
        "--arms",
        help="explicit arms instead of a suite: comma-separated analyst sets, each a "
             "'+'-joined list or 'all' (e.g. 'all,market,market+news'). The first is "
             "the reference",
    )
    parser.add_argument(
        "--universe", default=DEFAULT_UNIVERSE,
        help=f"named universe ({', '.join(sorted(UNIVERSES))}) or a comma-separated ticker list",
    )
    parser.add_argument("--holding-days", type=int, default=21, help="trading days held (default 21)")
    parser.add_argument("--step-days", type=int, default=21, help="calendar days between decisions")
    parser.add_argument("--benchmark", default="SPY", help="alpha benchmark (default SPY)")
    parser.add_argument(
        "--position-map", default="signed", choices=("signed", "long_only"),
        help="signed treats Sell as a short; long_only floors negatives at zero",
    )
    parser.add_argument("--entry-offset", type=int, default=1, help="bars between decision and entry")
    parser.add_argument("--knowledge-cutoff", help="model training cutoff, YYYY-MM-DD")
    parser.add_argument("--prices-dir", help="load <SYMBOL>.csv from here instead of yfinance")
    parser.add_argument(
        "--cache",
        help="shared JSONL decision cache. Safe across arms — names encode the arm's "
             "configuration — and strongly recommended: an ablation is the most "
             "expensive thing here",
    )
    parser.add_argument("--out", help="write the full markdown report here")
    parser.add_argument(
        "--html", help="write a charted, self-contained HTML report here"
    )
    parser.add_argument("--seed", type=int, default=0, help="bootstrap seed (default 0)")
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000, help="bootstrap resamples")
    parser.add_argument(
        "--dry-run", action="store_true", help="print the arms and total cost, then exit"
    )
    parser.add_argument("--verbose", action="store_true", help="debug logging and per-decision progress")
    return parser


def parse_arms(spec: str) -> list[AblationArm]:
    """Parse ``--arms all,market,market+news`` into arms; the first is the reference."""
    arms: list[AblationArm] = []
    for chunk in spec.split(","):
        chunk = chunk.strip().lower()
        if not chunk:
            continue
        if chunk == "all":
            arms.append(AblationArm(label="full pipeline"))
            continue
        analysts = tuple(a.strip() for a in chunk.split("+") if a.strip())
        arms.append(AblationArm(label=" + ".join(analysts), selected_analysts=analysts))
    if not arms:
        raise ValueError(f"no arms parsed from {spec!r}")
    return arms


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        arms = parse_arms(args.arms) if args.arms else resolve_suite(args.suite)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        grid = build_grid(
            resolve_universe(args.universe),
            start=args.start,
            end=args.end,
            holding_days=args.holding_days,
            step_days=args.step_days,
            knowledge_cutoff=args.knowledge_cutoff,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print(
            f"hint: with a {args.holding_days}-day hold, the latest usable decision date "
            f"is {latest_settled_date(args.holding_days)}",
            file=sys.stderr,
        )
        return 2

    if args.step_days < args.holding_days:
        print(
            f"warning: step-days ({args.step_days}) < holding-days ({args.holding_days}); "
            "holding windows overlap, so the confidence intervals will be optimistic.",
            file=sys.stderr,
        )

    if args.dry_run:
        return _dry_run(grid, arms)

    if not args.cache:
        print(
            "warning: no --cache given. An ablation runs every arm over the whole grid; "
            "without a cache an interruption loses all of it.",
            file=sys.stderr,
        )

    price_cache = None
    if args.prices_dir:
        price_cache = PriceCache()
        price_cache.load_csv_dir(args.prices_dir, [*grid.tickers, args.benchmark])
        for symbol, reason in price_cache.failures().items():
            print(f"warning: {symbol}: {reason}", file=sys.stderr)

    harness = Backtest(
        grid,
        price_cache=price_cache,
        benchmark=args.benchmark,
        position_map=args.position_map,
        entry_offset=args.entry_offset,
        cache_path=args.cache,
        bootstrap_iterations=args.bootstrap_iterations,
        seed=args.seed,
    )

    # Imported here so --dry-run and --help never touch LLM client construction.
    from .ablation import run_ablation

    try:
        result = run_ablation(
            grid, arms, backtest=harness, progress=_progress if args.verbose else None
        )
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(render_ablation_summary(result))

    if args.out:
        out_path = Path(args.out).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(render_ablation_report(result), encoding="utf-8")
        print(f"\nFull report written to {out_path}")

    if args.html:
        from tradingagents.webreport import render_ablation_html

        html_path = Path(args.html).expanduser()
        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.write_text(render_ablation_html(result), encoding="utf-8")
        print(f"HTML report written to {html_path}")

    return 0


def _dry_run(grid, arms) -> int:
    unique = dedupe_arms(arms)
    total = estimated_runs(grid, arms)
    print(f"Grid: {len(grid)} points ({len(grid.tickers)} tickers x {len(grid.dates)} dates)")
    print(f"  dates: {grid.start} .. {grid.end}")
    if grid.knowledge_cutoff:
        print(
            f"  cutoff: {grid.knowledge_cutoff} -> {grid.contaminated_count} contaminated, "
            f"{len(grid.clean())} out-of-sample"
        )
    print(f"\nArms ({len(unique)}), reference first:")
    for i, arm in enumerate(unique):
        marker = "  * " if i == 0 else "    "
        print(f"{marker}{arm.label}  ->  {arm.name}")
    if len(unique) < len(arms):
        print(f"\n  ({len(arms) - len(unique)} duplicate arm(s) dropped)")
    print(f"\nEstimated cost: {total} full multi-agent runs ({len(unique)} arms x {len(grid)} points).")
    print("Cached decisions from earlier runs are reused, so the real cost may be lower.")
    print("Pass --cache before starting; an ablation is the most expensive run here.")
    return 0


def _progress(strategy_name: str, point, rating: str) -> None:
    print(f"  [{strategy_name}] {point.key} -> {rating}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
