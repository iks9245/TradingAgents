"""Public agent factories, loaded only when their attributes are requested."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "AgentState",
    "create_msg_delete",
    "InvestDebateState",
    "RiskDebateState",
    "create_bear_researcher",
    "create_bull_researcher",
    "create_research_manager",
    "create_fundamentals_analyst",
    "create_market_analyst",
    "create_neutral_debator",
    "create_news_analyst",
    "create_aggressive_debator",
    "create_portfolio_manager",
    "create_conservative_debator",
    "create_sentiment_analyst",
    "create_social_media_analyst",  # deprecated; will be removed in a future version
    "create_trader",
]

# Keeping the package facade cheap lets utility modules remain usable when the
# optional LLM stack is unavailable.
_EXPORT_MODULES = {
    "AgentState": ".utils.agent_states",
    "create_msg_delete": ".utils.agent_utils",
    "InvestDebateState": ".utils.agent_states",
    "RiskDebateState": ".utils.agent_states",
    "create_bear_researcher": ".researchers.bear_researcher",
    "create_bull_researcher": ".researchers.bull_researcher",
    "create_research_manager": ".managers.research_manager",
    "create_fundamentals_analyst": ".analysts.fundamentals_analyst",
    "create_market_analyst": ".analysts.market_analyst",
    "create_neutral_debator": ".risk_mgmt.neutral_debator",
    "create_news_analyst": ".analysts.news_analyst",
    "create_aggressive_debator": ".risk_mgmt.aggressive_debator",
    "create_portfolio_manager": ".managers.portfolio_manager",
    "create_conservative_debator": ".risk_mgmt.conservative_debator",
    "create_sentiment_analyst": ".analysts.sentiment_analyst",
    "create_social_media_analyst": ".analysts.sentiment_analyst",
    "create_trader": ".trader.trader",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
