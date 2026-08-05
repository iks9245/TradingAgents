"""Strategies under test and the baselines they are measured against.

A strategy maps an :class:`~tradingagents.backtest.grid.EvalPoint` to one of the
five ratings in :data:`~tradingagents.agents.utils.rating.RATINGS_5_TIER`. The
harness then converts that rating to a position weight and scores it against the
realized forward return.

The baselines are the point of the module. Three of them, in increasing order of
how much they demand from the pipeline:

``AlwaysRating("Buy")``
    Buy-and-hold. Beating it means the pipeline earns its cost over doing nothing.

``UniformRandomRating``
    Coin-flip ratings. Failing to beat *this* means the pipeline carries no
    information at all.

``ShuffledRating``
    The sharp one. It replays the strategy's own rating distribution in random
    order, so it holds average bullishness fixed and randomizes only *which*
    point got *which* rating. A pipeline that beats buy-and-hold merely by being
    less-than-fully-long in a down market will match this baseline exactly; only
    a pipeline whose ratings are correctly *assigned* beats it.
"""

from __future__ import annotations

import logging
import random
from collections import Counter
from typing import Protocol

from tradingagents.agents.utils.rating import RATINGS_5_TIER
from tradingagents.dataflows.symbol_utils import crypto_base

from .grid import EvalPoint

logger = logging.getLogger(__name__)

# Rating -> position weight. Signed by default: Sell means short. A long-only
# desk should use LONG_ONLY_POSITIONS instead, which is a materially different
# (and usually more flattering, in a rising market) experiment.
SIGNED_POSITIONS: dict[str, float] = {
    "Buy": 1.0,
    "Overweight": 0.5,
    "Hold": 0.0,
    "Underweight": -0.5,
    "Sell": -1.0,
}

LONG_ONLY_POSITIONS: dict[str, float] = {
    "Buy": 1.0,
    "Overweight": 0.5,
    "Hold": 0.0,
    "Underweight": 0.0,
    "Sell": 0.0,
}

POSITION_MAPS = {"signed": SIGNED_POSITIONS, "long_only": LONG_ONLY_POSITIONS}


class Strategy(Protocol):
    """Anything that can rate an evaluation point."""

    name: str

    def decide(self, point: EvalPoint) -> str:
        """Return a 5-tier rating for ``point``."""
        ...


class AlwaysRating:
    """Emit one fixed rating everywhere. ``"Buy"`` is the buy-and-hold baseline."""

    def __init__(self, rating: str = "Buy", name: str | None = None):
        if rating not in RATINGS_5_TIER:
            raise ValueError(f"unknown rating {rating!r}; expected one of {RATINGS_5_TIER}")
        self.rating = rating
        self.name = name or f"always_{rating.lower()}"

    def decide(self, point: EvalPoint) -> str:
        return self.rating


class UniformRandomRating:
    """Draw ratings uniformly at random from the 5-tier scale.

    Seeded per evaluation point rather than from a running stream, so the same
    seed reproduces the same rating for the same point regardless of the order
    points are visited or whether an earlier run was interrupted and resumed.
    """

    def __init__(self, seed: int = 0, name: str = "random_uniform"):
        self.seed = seed
        self.name = name

    def decide(self, point: EvalPoint) -> str:
        rng = random.Random(f"{self.seed}:{self.name}:{point.key}")
        return rng.choice(RATINGS_5_TIER)


class ShuffledRating:
    """Replay a fixed rating distribution in a point-deterministic random order.

    Built from an already-completed strategy's ratings via :meth:`from_ratings`,
    so it matches that strategy's bullish tilt exactly while destroying any
    association between rating and instrument/date.
    """

    def __init__(self, ratings, seed: int = 0, name: str = "random_matched"):
        pool = list(ratings)
        if not pool:
            raise ValueError("ShuffledRating requires a non-empty rating pool")
        unknown = set(pool) - set(RATINGS_5_TIER)
        if unknown:
            raise ValueError(f"unknown ratings in pool: {sorted(unknown)}")
        self._pool = pool
        self.seed = seed
        self.name = name

    @classmethod
    def from_ratings(cls, ratings, seed: int = 0, name: str = "random_matched") -> ShuffledRating:
        return cls(ratings, seed=seed, name=name)

    @property
    def distribution(self) -> dict[str, int]:
        return dict(Counter(self._pool))

    def decide(self, point: EvalPoint) -> str:
        # Sampling with replacement from the pool, keyed on the point. This
        # preserves the pool's rating *proportions* in expectation without
        # requiring the harness to visit points in any particular order.
        rng = random.Random(f"{self.seed}:{self.name}:{point.key}")
        return rng.choice(self._pool)


class TradingAgentsStrategy:
    """Run the full multi-agent graph for each point and take the PM's rating.

    This is the only strategy that costs money. One ``propagate`` call per point
    means a few hundred thousand tokens each; the harness caches every decision
    to disk so an interrupted run resumes instead of re-spending.

    A single :class:`TradingAgentsGraph` is constructed lazily and reused across
    points, which is much cheaper than rebuilding LLM clients per call. The graph
    would otherwise accumulate memory-log state within a run, letting later
    decisions see reflections on earlier ones — realistic live, but a forward
    leak across the grid. :func:`build_backtest_config` disables the memory log
    for exactly that reason.

    ``name`` must be distinct per configuration. The decision cache is keyed on
    (strategy name, point), so two differently-configured strategies sharing a
    name would silently serve each other's decisions. :class:`AblationArm
    <tradingagents.backtest.ablation.AblationArm>` derives names from the
    configuration to make that mistake impossible.
    """

    def __init__(
        self,
        config: dict | None = None,
        *,
        selected_analysts=None,
        debug: bool = False,
        name: str = "trading_agents",
    ):
        self._config = config
        self._selected_analysts = selected_analysts
        self._debug = debug
        self._graph = None
        self.name = name

    def _get_graph(self):
        if self._graph is None:
            # Imported lazily so the baselines-only path never constructs LLM
            # clients, and so a machine with no API keys can still run them.
            from tradingagents.graph.trading_graph import TradingAgentsGraph

            kwargs = {"debug": self._debug, "config": self._config}
            if self._selected_analysts is not None:
                kwargs["selected_analysts"] = self._selected_analysts
            self._graph = TradingAgentsGraph(**kwargs)
        return self._graph

    def decide(self, point: EvalPoint) -> str:
        asset_type = "crypto" if crypto_base(point.ticker) else "stock"
        _, decision = self._get_graph().propagate(
            point.ticker, point.date, asset_type=asset_type
        )
        return decision


def build_backtest_config(base: dict | None = None, **overrides) -> dict:
    """Config for backtest runs: same pipeline, minus cross-point state.

    Disables the memory log (otherwise decisions later in the grid are informed
    by realized outcomes of earlier ones — information a live run genuinely has,
    but which makes a backtest's later points non-comparable to its earlier ones)
    and disables checkpointing (the harness has its own resume mechanism, at
    decision granularity rather than node granularity).
    """
    from tradingagents.default_config import DEFAULT_CONFIG

    config = dict(base or DEFAULT_CONFIG)
    config["memory_log_path"] = None
    config["checkpoint_enabled"] = False
    config.update(overrides)
    return config
