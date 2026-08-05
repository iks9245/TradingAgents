"""Backtest harness for measuring whether the pipeline's decisions carry signal.

The framework's own README describes it as "a research scaffold... not a strategy
with a fixed, replicable return." This package is how you check that claim on
your own configuration: it runs the multi-agent graph over a fixed grid of
(ticker, date) points, scores each decision against the realized forward return,
and compares it — paired, on the identical grid — against baselines that cost
nothing to run.

Quick start (no LLM calls, no API key needed)::

    python -m tradingagents.backtest --start 2024-01-01 --baselines-only

Programmatic use::

    from tradingagents.backtest import Backtest, build_grid, AlwaysRating
    from tradingagents.backtest import TradingAgentsStrategy, build_backtest_config

    grid = build_grid(["AAPL", "MSFT"], start="2025-01-01", holding_days=21)
    result = Backtest(grid, cache_path="runs/bt.jsonl").run(
        TradingAgentsStrategy(config=build_backtest_config()),
        [AlwaysRating("Buy")],
    )
    print(result.metrics["trading_agents"].mean_alpha)
"""

from .grid import EvalPoint, EvaluationGrid, build_grid, latest_settled_date
from .harness import Backtest, BacktestResult, DecisionCache
from .metrics import Decision, PairedComparison, StrategyMetrics, compare, spearman, summarize
from .prices import ForwardReturn, MissingPriceData, PriceCache
from .report import render_report, render_summary
from .strategies import (
    LONG_ONLY_POSITIONS,
    SIGNED_POSITIONS,
    AlwaysRating,
    ShuffledRating,
    TradingAgentsStrategy,
    UniformRandomRating,
    build_backtest_config,
)
from .universe import UNIVERSES, resolve_universe

__all__ = [
    "UNIVERSES",
    "AlwaysRating",
    "Backtest",
    "BacktestResult",
    "Decision",
    "DecisionCache",
    "EvalPoint",
    "EvaluationGrid",
    "ForwardReturn",
    "LONG_ONLY_POSITIONS",
    "MissingPriceData",
    "PairedComparison",
    "PriceCache",
    "SIGNED_POSITIONS",
    "ShuffledRating",
    "StrategyMetrics",
    "TradingAgentsStrategy",
    "UniformRandomRating",
    "build_backtest_config",
    "build_grid",
    "compare",
    "latest_settled_date",
    "render_report",
    "render_summary",
    "resolve_universe",
    "spearman",
    "summarize",
]
