"""Named ticker universes for reproducible backtests.

A universe is fixed and checked in so two runs of the same backtest evaluate the
same instruments. These lists are deliberately *not* index memberships pulled at
runtime: a live S&P 500 constituent list applied to a 2022 window is survivorship
bias, which inflates every long-biased strategy and would make the framework look
better than it is.

They still carry the milder form of the same bias — these are names that exist and
are liquid today — so absolute returns from any of them are optimistic. The
harness is built around *paired* comparison against baselines evaluated on the
identical grid, which is unaffected: survivorship lifts the strategy and its
baselines equally.
"""

from __future__ import annotations

# Large-cap US tech. Long-history names with dense news/fundamentals coverage,
# so every analyst has real data to work with rather than empty tool results.
MEGACAP_TECH = (
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "AVGO", "ORCL",
)

# Deliberately boring, low-beta names. Included so a backtest cannot pass on
# beta alone: a strategy that is simply long high-momentum tech will not look
# any better than buy-and-hold here.
DEFENSIVE = (
    "JNJ", "PG", "KO", "PEP", "WMT", "MRK", "VZ", "CVX",
)

# Sector ETFs. Diversified enough that idiosyncratic single-name news does not
# dominate, which makes them the cleanest read on whether the pipeline adds
# anything at the macro/sector level.
SECTOR_ETFS = (
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU",
)

# Liquid crypto, for exercising the asset_type="crypto" pipeline. Trades 7 days
# a week, so entry/exit bars are located on the crypto calendar and the equity
# benchmark is sampled at the nearest prior bar (see prices.forward_return).
CRYPTO = (
    "BTC-USD", "ETH-USD", "SOL-USD",
)

UNIVERSES: dict[str, tuple[str, ...]] = {
    "megacap_tech": MEGACAP_TECH,
    "defensive": DEFENSIVE,
    "sector_etfs": SECTOR_ETFS,
    "crypto": CRYPTO,
    # The default: spans growth, value, and sector exposure so results are not
    # an artifact of one regime favouring one style.
    "mixed": MEGACAP_TECH[:4] + DEFENSIVE[:4] + SECTOR_ETFS[:4],
    "smoke": ("AAPL", "XLK"),
}

DEFAULT_UNIVERSE = "mixed"


def resolve_universe(name_or_tickers: str | list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """Resolve a universe name, or a comma-separated/explicit ticker list.

    Accepts ``"mixed"``, ``"AAPL,MSFT"``, or ``["AAPL", "MSFT"]`` so the CLI and
    programmatic callers share one entry point.
    """
    if isinstance(name_or_tickers, (list, tuple)):
        tickers = tuple(str(t).strip().upper() for t in name_or_tickers if str(t).strip())
    elif name_or_tickers in UNIVERSES:
        return UNIVERSES[name_or_tickers]
    else:
        tickers = tuple(
            part.strip().upper() for part in str(name_or_tickers).split(",") if part.strip()
        )

    if not tickers:
        raise ValueError(
            f"empty universe: {name_or_tickers!r} is not a known universe "
            f"({', '.join(sorted(UNIVERSES))}) and contains no tickers"
        )
    return tickers
