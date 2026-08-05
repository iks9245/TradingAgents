"""CLI: ``python -m tradingagents.webreport <source> [-o out.html]``.

Renders a finished analysis run for reading in a browser. Accepts a report
directory, a ``full_states_log_*.json``, or any markdown file, so runs from
before this existed render too.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .analysis import render_any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tradingagents.webreport",
        description="Render an analysis report as a self-contained HTML page.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  # a report directory written by save_reports() or the CLI\n"
            "  python -m tradingagents.webreport ~/.tradingagents/logs/reports/NVDA_20260805_120000\n\n"
            "  # an older run that only kept the state log\n"
            "  python -m tradingagents.webreport \\\n"
            "      ~/.tradingagents/logs/NVDA/TradingAgentsStrategy_logs/full_states_log_2026-08-05.json\n"
        ),
    )
    parser.add_argument(
        "source",
        help="report directory, full_states_log_*.json, or a markdown file",
    )
    parser.add_argument(
        "-o", "--out",
        help="output path (default: <source>.html beside the source)",
    )
    parser.add_argument("--title", help="page title (default: derived from the source)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = Path(args.source).expanduser()

    try:
        html = render_any(source, title=args.title)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    out = Path(args.out).expanduser() if args.out else source.with_suffix(".html")
    if out.is_dir() or (not args.out and source.is_dir()):
        out = source.parent / f"{source.name}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
