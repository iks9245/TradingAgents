"""Deterministic verification snapshot for company fundamentals.

Statement dumps leave an LLM to divide, select a period, and infer a unit at
once. That is how a TTM margin is pasted into a fiscal-year row and a 6.01%
debt/equity figure becomes "6.01x". This module keeps the statement line items
visible, computes the ratios in Python, and gives the analyst one source of
truth for every derived fundamental claim.
"""

from __future__ import annotations

import functools
from collections.abc import Iterable

import pandas as pd
import yfinance as yf

from tradingagents.dataflows.fundamental_units import (
    fmt_money,
    fmt_multiple,
    fmt_price,
    pct_change,
    safe_ratio,
)
from tradingagents.dataflows.stockstats_utils import (
    filter_financials_by_date,
    load_ohlcv,
    yf_retry,
)
from tradingagents.dataflows.symbol_utils import normalize_symbol


def _row(df: pd.DataFrame, *candidates: str) -> pd.Series | None:
    """Find a statement row despite yfinance label drift, without guessing.

    Exact labels take precedence. If Yahoo changes capitalization or adds a
    qualifier, try a case-insensitive exact match and then a substring match
    for each requested label in order. A missing row stays missing: borrowing a
    nearby line item would recreate the statement-mapping errors this snapshot
    is intended to prevent.
    """
    if df is None or df.empty:
        return None
    labels = [str(label) for label in df.index]
    for candidate in candidates:
        if candidate in df.index:
            return df.loc[candidate]
    lowered = [(label, label.casefold()) for label in labels]
    for candidate in candidates:
        wanted = candidate.casefold()
        for label, comparable in lowered:
            if comparable == wanted:
                return df.loc[label]
    for candidate in candidates:
        wanted = candidate.casefold()
        for label, comparable in lowered:
            if wanted in comparable:
                return df.loc[label]
    return None


def _statement(ticker: yf.Ticker, name: str, curr_date: str) -> pd.DataFrame:
    """Read one statement safely and always remove future filing columns."""
    try:
        # Property access is rate-limited too; a retry here avoids a partial
        # snapshot when Yahoo returns HTTP 429 while loading one statement.
        data = yf_retry(lambda: getattr(ticker, name))
    except Exception:  # noqa: BLE001 -- one unavailable statement must not hide the others
        data = pd.DataFrame()
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame()
    return filter_financials_by_date(data, curr_date)


def _periods(*series: pd.Series | None, limit: int = 4) -> list[pd.Timestamp]:
    """Return dated statement columns newest first, limited for readable tables."""
    dates: set[pd.Timestamp] = set()
    for values in series:
        if values is None:
            continue
        for column in values.index:
            date = pd.to_datetime(column, errors="coerce")
            if not pd.isna(date):
                dates.add(pd.Timestamp(date))
    return sorted(dates, reverse=True)[:limit]


def _value(values: pd.Series | None, period: pd.Timestamp) -> float | None:
    """Return a numeric value for one fiscal column, treating NaN as missing."""
    if values is None:
        return None
    for column, raw in values.items():
        if pd.to_datetime(column, errors="coerce") == period:
            try:
                number = float(raw)
            except (TypeError, ValueError):
                return None
            return None if pd.isna(number) else number
    return None


def _millions(value: float | None) -> str:
    """Show statement arithmetic in millions, the unit used by the tables."""
    if value is None:
        return "N/A"
    return f"{value / 1_000_000:,.0f}"


def _eps(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2f}"


def _ratio_percent(numerator: float | None, denominator: float | None) -> str:
    ratio = safe_ratio(numerator, denominator)
    return "N/A" if ratio is None else f"{ratio * 100:.2f}%"


def _multiple(value: float | None) -> str:
    """Use the shared multiple formatter while keeping table cells compact."""
    rendered = fmt_multiple(value)
    return "N/A" if rendered is None else rendered.split("  [unit:", 1)[0]


def _growth(new: float | None, old: float | None, *, money: bool) -> str:
    """Render YoY growth with its operands so periods cannot be mixed up."""
    change = pct_change(new, old)
    if change is None:
        return "N/M"
    display = _millions if money else _eps
    return f"{change:.2f}%  (= {display(new)} / {display(old)} - 1)"


def _financial_currency(ticker: yf.Ticker) -> str:
    """Currency the statements are reported in.

    ``financialCurrency`` can differ from the trading currency (a US-listed ADR
    reporting in EUR), so naming it explicitly stops a reader from assuming USD
    and comparing figures across two currencies as if they were one.
    """
    try:
        info = yf_retry(lambda: ticker.info) or {}
    except Exception:  # noqa: BLE001 -- the snapshot is still valid without a currency label
        return "reporting currency"
    return info.get("financialCurrency") or info.get("currency") or "reporting currency"


def _vendor_info(ticker: yf.Ticker) -> dict:
    """The vendor's summary dict, or an empty one when it is unavailable."""
    try:
        return yf_retry(lambda: ticker.info) or {}
    except Exception:  # noqa: BLE001 -- the ratio cross-check is optional
        return {}


def _latest_close(symbol: str, curr_date: str) -> float | None:
    """Return the last close on or before the requested date, never inventing one."""
    try:
        data = load_ohlcv(symbol, curr_date)
    except Exception:  # noqa: BLE001 -- valuation is optional when price data is unavailable
        return None
    if data is None or data.empty or "Close" not in data.columns:
        return None
    frame = data.copy()
    if "Date" in frame.columns:
        dates = pd.to_datetime(frame["Date"], errors="coerce")
        frame = frame.loc[dates <= pd.Timestamp(curr_date)]
    if frame.empty:
        return None
    try:
        close = float(frame.iloc[-1]["Close"])
    except (TypeError, ValueError):
        return None
    return None if pd.isna(close) else close


# Free cash flow has no single definition, and the gap between them is not a
# rounding detail. Intel's 2026 Q2 simplified FCF is +4,450M while the company's
# own adjusted figure is -8,419M, because Intel's definition also carries partner
# contributions, government incentives and finance-lease payments. A report that
# shows only the positive number invites the conclusion that the fabs are
# self-funding. The vendor does not publish the company's definition, so the
# honest move is to name ours precisely and say the other exists.
_FCF_DEFINITION_NOTE = (
    "> **What \"free cash flow\" means here.** Every FCF figure above is the "
    "simplified definition: operating cash flow minus reported cash capital "
    "expenditure. A company's own \"adjusted free cash flow\" often differs "
    "materially — it may also deduct finance-lease payments and add partner "
    "contributions or government incentives, and the two can carry opposite "
    "signs in the same quarter. That figure comes from the filings and is not "
    "available here. Call this one simplified free cash flow, and do not present "
    "it as the company's own measure or as evidence that capital spending is "
    "self-funded."
)


def _append_income_sections(lines: list[str], annual: pd.DataFrame, currency: str) -> pd.Series | None:
    revenue = _row(annual, "Total Revenue", "Operating Revenue")
    gross_profit = _row(annual, "Gross Profit")
    operating_income = _row(annual, "Operating Income", "Total Operating Income As Reported")
    net_income = _row(annual, "Net Income", "Net Income Common Stockholders")
    diluted_eps = _row(annual, "Diluted EPS")
    basic_eps = _row(annual, "Basic EPS")
    periods = _periods(revenue, gross_profit, operating_income, net_income, diluted_eps, basic_eps)
    if not periods:
        return diluted_eps

    lines += ["", "### Annual income statement", "", f"Units: millions of {currency}, except per-share EPS.", "",
              "| Fiscal year end | Revenue | Gross profit | Operating income | Net income | Diluted EPS |",
              "|---|---:|---:|---:|---:|---:|"]
    for period in periods:
        lines.append(
            f"| {period:%Y-%m-%d} | {_millions(_value(revenue, period))} | "
            f"{_millions(_value(gross_profit, period))} | {_millions(_value(operating_income, period))} | "
            f"{_millions(_value(net_income, period))} | {_eps(_value(diluted_eps, period))} |"
        )

    margin_rows: list[str] = []
    for period in periods:
        rev = _value(revenue, period)
        cells = []
        for measure in (gross_profit, operating_income, net_income):
            amount = _value(measure, period)
            percent = _ratio_percent(amount, rev)
            cells.append("N/A" if percent == "N/A" else f"{percent}  ({_millions(amount)} / {_millions(rev)})")
        if any(cell != "N/A" for cell in cells):
            margin_rows.append(f"| {period:%Y-%m-%d} | " + " | ".join(cells) + " |")
    if margin_rows:
        lines += ["", "### Annual income-statement margins", "", f"Units in arithmetic: millions of {currency}.", "",
                  "| Fiscal year end | Gross margin | Operating margin | Net margin |",
                  "|---|---:|---:|---:|"] + margin_rows

    growth_rows: list[str] = []
    for index, period in enumerate(periods[:-1]):
        older = periods[index + 1]
        growth_rows.append(
            f"| {period:%Y-%m-%d} vs {older:%Y-%m-%d} | "
            f"{_growth(_value(revenue, period), _value(revenue, older), money=True)} | "
            f"{_growth(_value(operating_income, period), _value(operating_income, older), money=True)} | "
            f"{_growth(_value(net_income, period), _value(net_income, older), money=True)} | "
            f"{_growth(_value(diluted_eps, period), _value(diluted_eps, older), money=False)} |"
        )
    if growth_rows:
        lines += ["", "### Annual YoY growth", "", "| Fiscal years compared | Revenue | Operating income | Net income | Diluted EPS |",
                  "|---|---:|---:|---:|---:|"] + growth_rows
    return diluted_eps


def _income_components(df: pd.DataFrame) -> dict[str, pd.Series | None]:
    """The income-statement rows needed to rebuild operating income."""
    return {
        "revenue": _row(df, "Total Revenue", "Operating Revenue"),
        "gross_profit": _row(df, "Gross Profit"),
        "operating_income": _row(df, "Operating Income", "Total Operating Income As Reported"),
        "rd": _row(df, "Research And Development"),
        "sga": _row(df, "Selling General And Administration"),
        "restructuring": _row(df, "Restructuring And Mergern Acquisition", "Restructuring And Merger Acquisition"),
        "amortization": _row(df, "Amortization Of Intangibles Income Statement"),
        "other_opex": _row(df, "Other Operating Expenses"),
        "net_income": _row(df, "Net Income", "Net Income Common Stockholders"),
        "diluted_eps": _row(df, "Diluted EPS"),
    }


def _recomputed_operating_income(parts: dict, period: pd.Timestamp) -> tuple[float | None, float | None]:
    """Rebuild operating income from the expense lines; return (recomputed, opex_total)."""
    gross = _value(parts["gross_profit"], period)
    if gross is None:
        return None, None
    opex = 0.0
    seen = False
    for key in ("rd", "sga", "restructuring", "amortization", "other_opex"):
        amount = _value(parts[key], period)
        if amount is not None:
            opex += abs(amount)
            seen = True
    if not seen:
        return None, None
    return gross - opex, opex


def _append_operating_income_crosscheck(
    lines: list[str], quarterly: pd.DataFrame, annual: pd.DataFrame, currency: str
) -> None:
    """Check the vendor's operating income against its own expense lines.

    yfinance reports INTC's 2026 Q2 operating income as 1,966M while its own
    itemised lines give 1,805M — the 161M restructuring charge is absent from
    the vendor's ``Total Expenses`` but present as its own row. That inflated
    figure reached a shipped report as "operating income swung to $1.97B", and
    the same omission turned a GAAP operating *loss* in Q1 into an apparent
    profit, which the bull, the research manager and the portfolio manager each
    cited as proof of a turnaround.

    This detects internal inconsistency only. When the vendor's own expense row
    is itself understated, no arithmetic over its numbers can recover the filed
    figure — the note below says so rather than implying the recomputed value is
    authoritative.
    """
    rows: list[str] = []
    mismatch_notes: list[str] = []
    for label, df, limit in (("Quarter", quarterly, 4), ("Fiscal year", annual, 2)):
        parts = _income_components(df)
        for period in _periods(parts["operating_income"], parts["gross_profit"], limit=limit):
            reported = _value(parts["operating_income"], period)
            recomputed, _ = _recomputed_operating_income(parts, period)
            if reported is None or recomputed is None:
                continue
            revenue = _value(parts["revenue"], period)
            difference = reported - recomputed
            # Scale the tolerance to revenue: an operating income near zero
            # would make a relative test fire on rounding alone.
            allowed = max(0.005 * abs(revenue), 1_000_000.0) if revenue else 1_000_000.0
            # One-sided on purpose. ``difference > 0`` means the vendor booked
            # LESS expense than the rows it itemises — its operating income is
            # inflated by something it listed and then ignored, which is the
            # defect worth reporting. ``difference < 0`` only means this code
            # does not know every expense row the vendor uses (AMD carries an
            # intangibles-amortization line that older versions of this check
            # missed), and flagging that would blame the vendor for a gap in our
            # own enumeration.
            ok = difference <= allowed
            status = "consistent" if ok else "⚠️ MISMATCH"
            rows.append(
                f"| {label} {period:%Y-%m-%d} | {_millions(reported)} | {_millions(recomputed)} | "
                f"{_millions(difference)} | {status} |"
            )
            if ok:
                continue
            restructuring = _value(parts["restructuring"], period)
            cause = ""
            if restructuring is not None and abs(abs(restructuring) - abs(difference)) <= allowed:
                cause = (
                    f" The gap equals the restructuring/M&A line ({_millions(abs(restructuring))}), "
                    f"which the vendor's operating income excludes."
                )
            mismatch_notes.append(
                f"- {label} {period:%Y-%m-%d}: reported {_millions(reported)} vs "
                f"{_millions(recomputed)} from the line items.{cause}"
            )

    if not rows:
        return

    lines += [
        "",
        "### Operating income cross-check",
        "",
        f"Units: millions of {currency}. Recomputed = gross profit − (R&D + SG&A + "
        "restructuring/M&A + other operating expenses).",
        "",
        "| Period | Vendor reported | Recomputed from line items | Difference | Status |",
        "|---|---:|---:|---:|---|",
    ] + rows

    if mismatch_notes:
        lines += [
            "",
            "**The vendor's operating income does not reconcile with its own statement lines.**",
        ] + mismatch_notes + [
            "",
            "Do not describe operating income or operating margin for a mismatched period as "
            "a GAAP figure, and do not build a turnaround narrative on it — state both values "
            "and say they disagree. Two limits of this check: it compares the vendor against "
            "itself, so if an expense line is understated at source the recomputed figure is "
            "wrong too; and it only reports income that looks too high, so a period marked "
            "consistent is not thereby confirmed. Neither number here is a substitute for the "
            "filed statement.",
        ]
    else:
        lines += [
            "",
            "Reported and recomputed operating income agree for every period shown.",
        ]


# Vendor ratio fields, and the statement rows that reproduce them.
_VENDOR_RATIOS: tuple[tuple[str, str, str], ...] = (
    ("operatingMargins", "Operating margin", "operating_income"),
    ("profitMargins", "Net margin", "net_income"),
    ("grossMargins", "Gross margin", "gross_profit"),
)

# How close a recomputed window must be, in percentage points, to be called a
# match for the vendor's figure.
_RATIO_MATCH_TOLERANCE_PP = 0.25


def _margin_pct(numerator: float | None, denominator: float | None) -> float | None:
    ratio = safe_ratio(numerator, denominator)
    return None if ratio is None else ratio * 100


def _append_quarterly_income_sections(
    lines: list[str], quarterly: pd.DataFrame, currency: str
) -> None:
    """Quarterly margins and their period-on-period change, in percentage points.

    Margins were only ever published annually here, so a quarter-on-quarter move
    had to be worked out in the analyst's head — and was: a shipped report called
    Intel's Q2 gross margin "up 2.3 percentage points" when 39.38% to 40.36% is
    up 0.98. Printing the change with its operands leaves nothing to derive.

    The distinction between points and percent is stated because it is the other
    half of the same mistake: a move from 39.38% to 40.36% is +0.98 points, and
    separately +2.5% in relative terms. Neither is 2.3.
    """
    parts = _income_components(quarterly)
    periods = _periods(parts["revenue"], parts["gross_profit"], limit=5)
    if len(periods) < 1:
        return

    lines += [
        "",
        "### Quarterly income statement",
        "",
        f"Units: millions of {currency}, except per-share EPS.",
        "",
        "| Quarter end | Revenue | Gross profit | Operating income | Net income | Diluted EPS |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for period in periods:
        lines.append(
            f"| {period:%Y-%m-%d} | {_millions(_value(parts['revenue'], period))} | "
            f"{_millions(_value(parts['gross_profit'], period))} | "
            f"{_millions(_value(parts['operating_income'], period))} | "
            f"{_millions(_value(parts['net_income'], period))} | "
            f"{_eps(_value(parts['diluted_eps'], period))} |"
        )

    measures = (
        ("Gross margin", "gross_profit"),
        ("Operating margin", "operating_income"),
        ("Net margin", "net_income"),
    )
    margins: dict[pd.Timestamp, dict[str, float | None]] = {}
    margin_rows: list[str] = []
    for period in periods:
        revenue = _value(parts["revenue"], period)
        cells, row = [], {}
        for label, key in measures:
            amount = _value(parts[key], period)
            value = _margin_pct(amount, revenue)
            row[label] = value
            cells.append(
                "N/A" if value is None
                else f"{value:.2f}%  ({_millions(amount)} / {_millions(revenue)})"
            )
        margins[period] = row
        if any(cell != "N/A" for cell in cells):
            margin_rows.append(f"| {period:%Y-%m-%d} | " + " | ".join(cells) + " |")
    if margin_rows:
        lines += [
            "",
            "### Quarterly margins",
            "",
            f"Units in arithmetic: millions of {currency}.",
            "",
            "| Quarter end | Gross margin | Operating margin | Net margin |",
            "|---|---:|---:|---:|",
        ] + margin_rows

    change_rows: list[str] = []
    for index, period in enumerate(periods[:-1]):
        older = periods[index + 1]
        cells = []
        for label, _key in measures:
            new, prior = margins[period].get(label), margins[older].get(label)
            if new is None or prior is None:
                cells.append("N/A")
                continue
            cells.append(f"{new - prior:+.2f} pp  ({new:.2f}% - {prior:.2f}%)")
        if any(cell != "N/A" for cell in cells):
            change_rows.append(
                f"| {period:%Y-%m-%d} vs {older:%Y-%m-%d} | " + " | ".join(cells) + " |"
            )
    if change_rows:
        lines += [
            "",
            "### Quarter-on-quarter margin change",
            "",
            "| Quarters compared | Gross margin | Operating margin | Net margin |",
            "|---|---:|---:|---:|",
        ] + change_rows + [
            "",
            "> Quote these changes in **percentage points**, with the sign and the two "
            "margins they came from. A margin moving from 39.38% to 40.36% rose 0.98 "
            "points — that is a different statement from \"rose 2.5%\" (the relative "
            "change) and neither of them is a number you should round up to a rougher "
            "one. Do not compare a quarterly margin against an annual or TTM margin: "
            "those are different windows and the difference is not a trend.",
        ]


def _append_vendor_ratio_crosscheck(
    lines: list[str], info: dict, quarterly: pd.DataFrame
) -> None:
    """Resolve which period each vendor ratio actually covers.

    yfinance publishes ``profitMargins`` and ``operatingMargins`` side by side
    in one flat dict, but they cover different windows: the profit margin is
    trailing-twelve-month while the operating margin is the most recent quarter
    alone. Checked against INTC, AAPL, MSFT, NVDA and TSM, each matches its own
    window exactly. Assuming a shared period is how Intel's single-quarter
    12.19% operating margin was reported as its trailing-twelve-month figure,
    when four quarters of its own statements give 7.55%.

    Rather than assert a period, this recomputes both windows from the
    statements and reports which one the vendor's number reproduces — and says
    so plainly when it reproduces neither.
    """
    if not isinstance(info, dict) or quarterly is None or quarterly.empty:
        return
    parts = _income_components(quarterly)
    periods = _periods(parts["revenue"], limit=4)
    if not periods:
        return

    def _window(measure: str, count: int) -> tuple[float | None, str]:
        used = periods[:count]
        revenue = [_value(parts["revenue"], p) for p in used]
        values = [_value(parts[measure], p) for p in used]
        if any(v is None for v in revenue + values) or not revenue:
            return None, "N/A"
        total_rev, total_val = sum(revenue), sum(values)
        ratio = safe_ratio(total_val, total_rev)
        if ratio is None:
            return None, "N/A"
        return ratio * 100, f"{ratio * 100:.2f}%  ({_millions(total_val)} / {_millions(total_rev)})"

    rows: list[str] = []
    unmatched: list[str] = []
    for field, label, measure in _VENDOR_RATIOS:
        raw = info.get(field)
        vendor = None if raw is None else safe_ratio(raw, 1)
        if vendor is None:
            continue
        vendor_pct = vendor * 100
        mrq, mrq_text = _window(measure, 1)
        ttm, ttm_text = _window(measure, 4)
        match = "⚠️ neither"
        if mrq is not None and abs(vendor_pct - mrq) <= _RATIO_MATCH_TOLERANCE_PP:
            match = "most recent quarter"
        elif ttm is not None and abs(vendor_pct - ttm) <= _RATIO_MATCH_TOLERANCE_PP:
            match = "trailing 12 months"
        else:
            unmatched.append(f"- `{field}` ({vendor_pct:.2f}%) matches neither window.")
        rows.append(
            f"| {label} (`{field}`) | {vendor_pct:.2f}% | {match} | {mrq_text} | {ttm_text} |"
        )

    if not rows:
        return

    lines += [
        "",
        "### Vendor ratio cross-check",
        "",
        f"The vendor publishes these ratios without a period. Recomputed here from the "
        f"statements over both windows; the matching column says which one each number "
        f"actually is. Latest quarter used: {periods[0]:%Y-%m-%d}.",
        "",
        "| Vendor field | Vendor value | Period it matches | Most recent quarter | Trailing 12 months |",
        "|---|---:|---|---:|---:|",
    ] + rows + [
        "",
        "Quote a vendor ratio only with the period named in the matching column. These "
        "windows are not interchangeable: a single strong quarter and a trailing year "
        "can point in opposite directions.",
    ]
    if unmatched:
        lines += [
            "",
            "**A vendor ratio does not reproduce either window from the statements.**",
        ] + unmatched + [
            "Treat it as unverified: state the statement-derived figure and its window "
            "instead, and do not attach a period to the vendor's number.",
        ]


# Statement rows whose name promises a benefit. A negative value under one of
# these is a loss wearing a gain's label, and a reader who trusts the label
# inverts the sign: Intel's 2026 Q2 "Gain On Sale Of Security" holds -12.476B,
# and a shipped report published it as a 12.43B non-cash *gain*, then used it to
# explain away the quarter's net loss.
_GAIN_WORDS = ("gain",)


def _append_sign_label_check(lines: list[str], quarterly: pd.DataFrame, currency: str) -> None:
    """Flag statement rows whose label implies a gain but whose value is negative."""
    if quarterly is None or quarterly.empty:
        return
    findings: list[str] = []
    for row_label in quarterly.index:
        name = str(row_label)
        if not any(word in name.casefold() for word in _GAIN_WORDS):
            continue
        series = quarterly.loc[row_label]
        if isinstance(series, pd.DataFrame):  # duplicated label; not worth guessing
            continue
        for period in _periods(series, limit=4):
            value = _value(series, period)
            if value is None or value >= 0:
                continue
            findings.append(
                f"| {period:%Y-%m-%d} | {name} | {_millions(value)} | "
                f"a LOSS of {_millions(abs(value))} |"
            )
    if not findings:
        return
    lines += [
        "",
        "### Sign-versus-label contradictions",
        "",
        f"Units: millions of {currency}. These rows are named as gains but hold negative "
        "values, which means the opposite of what the label says.",
        "",
        "| Period | Row label | Value as stored | What it actually is |",
        "|---|---|---:|---|",
    ] + findings + [
        "",
        "Report each of these as the loss it is. Do not describe a negative row named "
        "\"Gain ...\" as a gain, and do not cite one as a benign or offsetting item when "
        "explaining a net loss — it is part of the loss.",
    ]


def _append_balance_section(lines: list[str], quarterly: pd.DataFrame, currency: str) -> None:
    assets = _row(quarterly, "Total Assets")
    liabilities = _row(quarterly, "Total Liabilities Net Minority Interest", "Total Liabilities")
    equity = _row(quarterly, "Stockholders Equity", "Total Equity Gross Minority Interest")
    total_debt = _row(quarterly, "Total Debt")
    long_debt = _row(quarterly, "Long Term Debt")
    cash = _row(quarterly, "Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments")
    current_assets = _row(quarterly, "Current Assets", "Total Current Assets")
    current_liabilities = _row(quarterly, "Current Liabilities", "Total Current Liabilities")
    inventory = _row(quarterly, "Inventory")
    period_list = _periods(assets, liabilities, equity, total_debt, long_debt, cash, current_assets, current_liabilities, inventory, limit=1)
    if not period_list:
        return
    period = period_list[0]
    items = (
        ("Total assets", assets), ("Total liabilities", liabilities), ("Stockholders' equity", equity),
        ("Total debt", total_debt), ("Long-term debt", long_debt), ("Cash and cash equivalents", cash),
        ("Current assets", current_assets), ("Current liabilities", current_liabilities), ("Inventory", inventory),
    )
    present = [(label, _value(values, period)) for label, values in items if _value(values, period) is not None]
    if not present:
        return
    lines += ["", f"### Balance sheet, most recent quarter ({period:%Y-%m-%d})", "", "| Line item | Value |", "|---|---:|"]
    for label, value in present:
        lines.append(f"| {label} | {fmt_money(value, currency) or 'N/A'} |")

    debt, long_term, eq = _value(total_debt, period), _value(long_debt, period), _value(equity, period)
    current, current_liab, inv = _value(current_assets, period), _value(current_liabilities, period), _value(inventory, period)
    ratio_rows: list[str] = []
    debt_ratio = safe_ratio(debt, eq)
    if debt_ratio is not None:
        ratio_rows.append(f"| Total debt / equity | {debt_ratio * 100:.2f}%  (= {debt_ratio:.4f}x)  ({_millions(debt)} / {_millions(eq)}) |")
    long_ratio = safe_ratio(long_term, eq)
    if long_ratio is not None:
        ratio_rows.append(f"| Long-term debt / equity | {long_ratio * 100:.2f}%  (= {long_ratio:.4f}x)  ({_millions(long_term)} / {_millions(eq)}) |")
    current_ratio = safe_ratio(current, current_liab)
    if current_ratio is not None:
        ratio_rows.append(f"| Current ratio | {_multiple(current_ratio)}  ({_millions(current)} / {_millions(current_liab)}) |")
    quick_ratio = safe_ratio(None if current is None or inv is None else current - inv, current_liab)
    if quick_ratio is not None:
        ratio_rows.append(f"| Quick ratio | {_multiple(quick_ratio)}  (({_millions(current)} - {_millions(inv)}) / {_millions(current_liab)}) |")
    if ratio_rows:
        lines += ["", "| Ratio | Verified calculation |", "|---|---:|"] + ratio_rows
        lines += ["", "Debt/equity is a balance-sheet ratio of borrowings to equity; goodwill and intangibles are assets and do not raise it."]


def _append_cash_flow_section(lines: list[str], annual: pd.DataFrame, quarterly: pd.DataFrame, currency: str) -> None:
    annual_ocf = _row(annual, "Operating Cash Flow", "Total Cash From Operating Activities")
    annual_capex = _row(annual, "Capital Expenditure", "Capital Expenditures")
    annual_fcf = _row(annual, "Free Cash Flow")
    periods = _periods(annual_ocf, annual_capex, annual_fcf)
    if periods:
        lines += ["", "### Cash flow — annual", "", f"Capex is shown as spend using abs(yfinance capex), in millions of {currency}.", "",
                  "| Fiscal year end | Operating cash flow | Capex spend | Simplified FCF (OCF - capex) | OCF YoY | Capex YoY | FCF YoY |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for index, period in enumerate(periods):
            ocf, capex, fcf = _value(annual_ocf, period), _value(annual_capex, period), _value(annual_fcf, period)
            capex_spend = None if capex is None else abs(capex)
            if index + 1 < len(periods):
                older = periods[index + 1]
                old_ocf = _value(annual_ocf, older)
                old_capex = _value(annual_capex, older)
                old_fcf = _value(annual_fcf, older)
                growths = (_growth(ocf, old_ocf, money=True), _growth(capex_spend, None if old_capex is None else abs(old_capex), money=True), _growth(fcf, old_fcf, money=True))
            else:
                growths = ("N/A", "N/A", "N/A")
            lines.append(
                f"| {period:%Y-%m-%d} | {_millions(ocf)} | {_millions(capex_spend)} | {_millions(fcf)} | "
                + " | ".join(growths) + " |"
            )

    q_ocf = _row(quarterly, "Operating Cash Flow", "Total Cash From Operating Activities")
    q_capex = _row(quarterly, "Capital Expenditure", "Capital Expenditures")
    q_fcf = _row(quarterly, "Free Cash Flow")
    quarter_periods = _periods(q_ocf, q_capex, q_fcf, limit=100)
    if quarter_periods:
        latest = quarter_periods[0]
        comparison = next((date for date in quarter_periods if date.year == latest.year - 1 and date.month == latest.month), None)
        if comparison is not None:
            lines += ["", f"### Cash flow — quarterly YoY ({latest:%Y-%m-%d} vs {comparison:%Y-%m-%d})", "",
                      "| Metric | Most recent quarter | Same quarter one year earlier | YoY |", "|---|---:|---:|---:|"]
            for label, values, is_capex in (("Operating cash flow", q_ocf, False), ("Capex spend", q_capex, True), ("Simplified FCF (OCF - capex)", q_fcf, False)):
                latest_value, old_value = _value(values, latest), _value(values, comparison)
                if is_capex:
                    latest_value = None if latest_value is None else abs(latest_value)
                    old_value = None if old_value is None else abs(old_value)
                if latest_value is None and old_value is None:
                    continue
                lines.append(f"| {label} | {_millions(latest_value)} | {_millions(old_value)} | {_growth(latest_value, old_value, money=True)} |")

    # A table conveys "which column am I in" by position, and that information
    # does not survive being lifted into prose. On the 2026-08-06 INTC report
    # Q1's free cash flow of -2,540 was quoted as its *operating* cash flow by
    # the bear, then by the research manager, then by the portfolio manager —
    # who labelled it "verified". Both numbers were verified; the column was
    # not. These lines carry each figure's identity next to the figure, so a
    # copied fragment stays self-describing.
    labelled = []
    for period in _periods(q_ocf, q_capex, q_fcf, limit=5):
        ocf, capex, fcf = _value(q_ocf, period), _value(q_capex, period), _value(q_fcf, period)
        if ocf is None and fcf is None:
            continue
        capex_spend = None if capex is None else abs(capex)
        labelled.append(
            f"- {period:%Y-%m-%d}: OCF {_millions(ocf)} | capex {_millions(capex_spend)} "
            f"| simplified FCF (OCF - capex) {_millions(fcf)}"
        )
    if labelled:
        lines += [
            "",
            "### Cash flow — labelled series (quote these lines, not table columns)",
            "",
            f"Units: millions of {currency}. Every figure carries its own name, so a value "
            "lifted out of context cannot lose which measure it is.",
            "",
        ] + labelled + [
            "",
            "OCF and FCF are different measures and differ by capex. Never restate one as "
            "the other, and always name which you are quoting.",
        ]

    if periods or quarter_periods:
        lines += ["", "> Capex and free-cash-flow growth rates differ between the annual and quarterly views. State which one you are quoting, with the period. Do not describe a +83.5% quarterly change as \"doubled\"."]
        lines += ["", _FCF_DEFINITION_NOTE]


def _append_price_statistics_section(lines: list[str], symbol: str, curr_date: str, currency: str) -> None:
    """Restate the market snapshot's price statistics, from the same source.

    The fundamentals report needs a 50-day average and a 52-week range, and the
    vendor's ``info`` dict offers both — computed on its own schedule and
    adjustment basis. Taking them from there put a 50-day average of 512.95 in
    the fundamentals section of a report whose technical section said 514.33.
    These come from the same settled OHLCV frame the market snapshot uses, so
    the two sections cannot disagree.
    """
    # Imported here rather than at module scope: the market validator imports
    # nothing from this module today, and a top-level import would make that
    # relationship a cycle the first time it does.
    from tradingagents.dataflows.market_data_validator import get_trade_reference_levels

    ref = get_trade_reference_levels(symbol, curr_date)
    if ref is None:
        return
    rows = [
        ("Last close", ref.close),
        ("50-day SMA", ref.sma50),
        ("200-day SMA", ref.sma200),
        ("52-week high", ref.week52_high),
        ("52-week low", ref.week52_low),
        ("ATR (daily volatility)", ref.atr),
    ]
    present = [(label, value) for label, value in rows if value is not None]
    if not present:
        return
    lines += [
        "",
        f"### Price statistics (as of {ref.as_of}, bar status {ref.bar_status})",
        "",
        "| Statistic | Value |",
        "|---|---:|",
    ]
    for label, value in present:
        lines.append(f"| {label} | {fmt_price(value, currency) or 'N/A'} |")
    lines += [
        "",
        "These are computed from the same settled daily bars as the verified market "
        "snapshot, so both reports quote identical numbers. Do not substitute a "
        "vendor quote-feed moving average or 52-week figure for these — that is how "
        "one statistic ends up with two values in one report.",
    ]


def _append_valuation_section(lines: list[str], price: float | None, annual_eps: pd.Series | None, quarterly: pd.DataFrame, currency: str) -> None:
    if price is None:
        return
    annual_periods = _periods(annual_eps, limit=1)
    quarterly_eps = _row(quarterly, "Diluted EPS")
    quarter_periods = _periods(quarterly_eps)
    rows: list[tuple[str, float]] = []
    if annual_periods:
        annual_value = _value(annual_eps, annual_periods[0])
        if annual_value is not None and annual_value > 0:
            rows.append((f"FY{annual_periods[0].year} GAAP diluted", annual_value))
    if len(quarter_periods) >= 4:
        values = [_value(quarterly_eps, period) for period in quarter_periods[:4]]
        if all(value is not None for value in values):
            ttm_eps = sum(values)  # type: ignore[arg-type]
            if ttm_eps > 0:
                rows.append(("TTM GAAP diluted (sum of last 4 quarters)", ttm_eps))
    if not rows:
        return
    lines += ["", "### Valuation conversion reference", "", f"Reference price used: {fmt_price(price, currency) or 'N/A'}.", "",
              f"| EPS basis | EPS | P/E at {price:.2f} |", "|---|---:|---:|"]
    for basis, eps in rows:
        lines.append(f"| {basis} | {_eps(eps)} | {_multiple(safe_ratio(price, eps))} |")

    fy_row = next(((basis, eps) for basis, eps in rows if basis.startswith("FY")), None)
    if fy_row is not None:
        basis, eps = fy_row
        example_price = price * 0.85
        example_multiple = safe_ratio(example_price, eps)
        lines += ["", f"To state a P/E at any other price P, compute P / EPS_basis using a row above and name the basis. A price of {example_price:.2f} is {_multiple(example_multiple)} {basis} EPS — it is not \"about 40x 2025 PE\". Never quote a P/E without naming its EPS basis and period."]

    # The TTM row is summed from four quarterly diluted EPS figures, so it can
    # differ by a few cents from a vendor's own trailing EPS field (which
    # reweights the share count rather than adding rounded quarters). Saying so
    # keeps a reader from presenting the two as a contradiction — or from
    # quietly averaging them into a third number that matches no source.
    if any(basis.startswith("TTM") for basis, _ in rows):
        lines += [
            "",
            "The TTM row is the sum of the last four reported quarterly diluted EPS "
            "figures. A vendor's own trailing-EPS field may differ by a few cents "
            "because it reweights the diluted share count instead of adding rounded "
            "quarters. Both are legitimate; cite one, name which, and do not blend them.",
        ]


@functools.lru_cache(maxsize=64)
def render_fundamentals_snapshot_block(symbol: str, curr_date: str) -> str:
    """Build the snapshot for prompt injection, or an explicit unavailable notice.

    Offered as a tool, this snapshot was simply not called: on the 2026-08-06
    INTC run the analyst ignored the instruction and sourced every ratio from
    the raw vendor dump instead, so the recomputed margins, the P/E conversion
    table, and the printed arithmetic never reached the report. A prompt cannot
    make a model call a tool; pre-fetching removes the choice.

    Cached because the analyst node re-runs on every turn of its tool loop and
    the underlying filings do not change within a run.
    """
    if not symbol or not curr_date:
        return _UNAVAILABLE_NOTICE
    try:
        return build_verified_fundamentals_snapshot(symbol, curr_date)
    except Exception:  # noqa: BLE001 — a missing snapshot must not block the run
        return _UNAVAILABLE_NOTICE


_UNAVAILABLE_NOTICE = (
    "**Verified fundamentals snapshot: UNAVAILABLE.** No statement data could be "
    "resolved for this instrument. Do not state a margin, growth rate, leverage "
    "ratio, or valuation multiple that you derived yourself — say the figures "
    "could not be verified and report only what the raw statements show, with "
    "their periods named."
)


def build_verified_fundamentals_snapshot(
    symbol: str,
    curr_date: str,
    reference_price: float | None = None,
) -> str:
    """Render date-safe statement facts and deterministic fundamental ratios."""
    canonical = normalize_symbol(symbol)
    ticker = yf.Ticker(canonical)
    annual_income = _statement(ticker, "income_stmt", curr_date)
    quarterly_income = _statement(ticker, "quarterly_income_stmt", curr_date)
    annual_balance = _statement(ticker, "balance_sheet", curr_date)
    quarterly_balance = _statement(ticker, "quarterly_balance_sheet", curr_date)
    annual_cashflow = _statement(ticker, "cashflow", curr_date)
    quarterly_cashflow = _statement(ticker, "quarterly_cashflow", curr_date)
    statements: Iterable[pd.DataFrame] = (annual_income, quarterly_income, annual_balance, quarterly_balance, annual_cashflow, quarterly_cashflow)
    if not any(not statement.empty and len(statement.columns) for statement in statements):
        raise ValueError(f"No usable statement data available for {symbol}.")

    lines = [
        f"## Verified fundamentals snapshot for {symbol.upper()}",
        "",
        f"- Requested analysis date: {curr_date}",
        "- Statement columns dated after the requested date are excluded.",
        "- Every ratio below is computed in Python from the line items shown. The arithmetic is printed so it can be checked.",
    ]
    currency = _financial_currency(ticker)
    annual_eps = _append_income_sections(lines, annual_income, currency)
    _append_quarterly_income_sections(lines, quarterly_income, currency)
    _append_operating_income_crosscheck(lines, quarterly_income, annual_income, currency)
    _append_vendor_ratio_crosscheck(lines, _vendor_info(ticker), quarterly_income)
    _append_sign_label_check(lines, quarterly_income, currency)
    _append_balance_section(lines, quarterly_balance, currency)
    _append_cash_flow_section(lines, annual_cashflow, quarterly_cashflow, currency)
    _append_price_statistics_section(lines, symbol, curr_date, currency)
    price = reference_price if reference_price is not None else _latest_close(symbol, curr_date)
    _append_valuation_section(lines, price, annual_eps, quarterly_income, currency)
    lines += [
        "",
        "Use this snapshot as the source of truth for every fundamental ratio, growth rate, and valuation multiple. Quote these figures; do not recompute them and do not carry a ratio from any other tool output, news summary, or social-media post into this report as fact. If another source states a figure that conflicts with this snapshot, report the conflict and name both sources rather than reconciling them. Every percentage you cite must name its period and its basis.",
    ]
    return "\n".join(lines)
