from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import (
    get_balance_sheet,
    get_cashflow,
    get_fundamentals,
    get_income_statement,
    get_instrument_context_from_state,
    get_language_instruction,
)
from tradingagents.dataflows.fundamentals_validator import (
    render_fundamentals_snapshot_block,
)


def create_fundamentals_analyst(llm):
    def fundamentals_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = get_instrument_context_from_state(state)

        tools = [
            get_fundamentals,
            get_balance_sheet,
            get_cashflow,
            get_income_statement,
        ]

        # The snapshot used to be a tool the prompt asked for. On the
        # 2026-08-06 INTC run the model simply did not call it and sourced every
        # ratio from the raw vendor dump instead, so none of the recomputed
        # figures reached the report. Pre-fetching it into the prompt — the same
        # shape the sentiment analyst uses for its pre-fetched sources — removes
        # the option of skipping it.
        snapshot_block = render_fundamentals_snapshot_block(
            state["company_of_interest"], current_date
        )

        system_message = (
            "You are a researcher tasked with analyzing fundamental information over the past week about a company. Please write a comprehensive report of the company's fundamental information such as financial documents, company profile, basic company financials, and company financial history to gain a full view of the company's fundamental information to inform traders. Make sure to include as much detail as possible. Provide specific, actionable insights with supporting evidence to help traders make informed decisions."
            + " Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."
            + " Use the available tools: `get_fundamentals` for comprehensive company analysis, `get_balance_sheet`, `get_cashflow`, and `get_income_statement` for specific financial statements."
            # Derived fundamentals are where this agent has historically gone
            # wrong: dividing statement lines in-head, misreading a vendor's
            # percent as a multiple, and pasting a TTM ratio into a fiscal-year
            # row. The snapshot below computes those in Python and shows its
            # work, so the instruction is to quote it rather than to recompute.
            + "\n\n<start_of_verified_fundamentals>\n"
            + snapshot_block
            + "\n<end_of_verified_fundamentals>\n\n"
            + "The block above is already computed for you — there is no tool to call for it."
            " Treat it as the source of truth for every margin, growth rate, leverage ratio,"
            " liquidity ratio, and valuation multiple. Quote its figures rather than deriving"
            " your own: do not divide statement line items yourself, and do not restate a value"
            " in a different unit than the one it printed. Each figure you cite must carry the"
            " period given — a value labelled (TTM) or (MRQ) must never be presented as a"
            " fiscal-year figure, and a P/E must always name its EPS basis. If another tool"
            " output, news item, or social-media post states a number that conflicts with this"
            " block, report the conflict and name both sources instead of reconciling them."
            + get_language_instruction(),
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}."
                    " Today's date is {current_date}; treat it as 'now' for all analysis and tool-call date ranges. {instrument_context}\n"
                    "{system_message}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        chain = prompt | llm.bind_tools(tools)

        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "fundamentals_report": report,
        }

    return fundamentals_analyst_node
