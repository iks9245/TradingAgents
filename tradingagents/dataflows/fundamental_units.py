"""Unit-explicit rendering for vendor fundamental fields.

A vendor's fundamentals payload mixes three numeric conventions in one flat
namespace. yfinance's ``info`` is the worst offender:

    profitMargins   0.134   a FRACTION       -> 13.4%
    debtToEquity    6.01    a PERCENT        -> 6.01%, i.e. 0.06x
    currentRatio    2.73    a BARE MULTIPLE  -> 2.73x

Emitting those as bare numbers next to each other invites a 100x unit error:
``Debt to Equity: 6.01`` reads as "6.01x debt/equity" (debt is 601% of equity,
alarming) when it actually means 6.01% (debt is 6% of equity, conservative) —
an error that inverts the balance-sheet conclusion. The same flat rendering
also lets a trailing-twelve-month margin be pasted into a fiscal-year table,
because nothing in the line says which period it covers.

Every formatter here renders the unit inline and, where a value can honestly
be read two ways, renders both readings. Formatters never guess: a field whose
convention is genuinely ambiguous across vendor versions is rendered by
``fmt_raw`` with an explicit do-not-convert marker rather than silently picked.
"""

from __future__ import annotations

from collections.abc import Callable

# Magnitude suffixes for money formatting, largest first.
_MAGNITUDES: tuple[tuple[float, str], ...] = (
    (1e12, "T"),
    (1e9, "B"),
    (1e6, "M"),
    (1e3, "K"),
)


def _as_float(value) -> float | None:
    """Coerce a vendor value to float, or None when it is not a usable number."""
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    # NaN is not usable and compares false against itself.
    return None if result != result else result


def fmt_fraction_as_pct(value, raw_field: str | None = None) -> str | None:
    """Render a fraction (0.134) as a percent, keeping the raw value visible.

    Used for yfinance margin/return fields, which are always fractions.
    """
    number = _as_float(value)
    if number is None:
        return None
    suffix = f", raw {number:.4f}" if raw_field is None else f", raw {number:.4f} from `{raw_field}`"
    return f"{number * 100:.2f}%  [unit: percent{suffix}]"


def fmt_percent_with_multiple(value, raw_field: str | None = None) -> str | None:
    """Render a value the vendor already expresses in percent, plus its multiple.

    yfinance reports ``debtToEquity`` in percent (6.01 means 6.01%), so a reader
    who takes it as a multiple is off by 100x. Rendering both readings removes
    the ambiguity at the point of use.
    """
    number = _as_float(value)
    if number is None:
        return None
    field = f" `{raw_field}`" if raw_field else ""
    return (
        f"{number:.2f}%  (= {number / 100:.4f}x)  "
        f"[unit: PERCENT — vendor field{field} is already a percentage, "
        f"NOT a multiple; do not quote {number:.2f} as '{number:.2f}x']"
    )


def fmt_multiple(value) -> str | None:
    """Render a bare ratio/multiple (current ratio, P/E, P/B)."""
    number = _as_float(value)
    if number is None:
        return None
    return f"{number:.2f}x  [unit: multiple]"


def fmt_money(value, currency: str = "USD") -> str | None:
    """Render a currency amount with a magnitude suffix and the exact figure."""
    number = _as_float(value)
    if number is None:
        return None
    magnitude = abs(number)
    for threshold, suffix in _MAGNITUDES:
        if magnitude >= threshold:
            return f"{number / threshold:,.2f}{suffix} {currency}  (exact: {number:,.0f})"
    return f"{number:,.2f} {currency}"


def fmt_price(value, currency: str = "USD") -> str | None:
    """Render a per-share price or per-share amount."""
    number = _as_float(value)
    if number is None:
        return None
    return f"{number:,.2f} {currency}"


def fmt_plain(value) -> str | None:
    """Render a unitless number (beta) without implying a unit."""
    number = _as_float(value)
    if number is None:
        return None
    return f"{number:.2f}  [unitless]"


def fmt_raw(value, raw_field: str | None = None) -> str | None:
    """Render a value whose unit convention is not reliably known.

    Some vendor fields have changed convention between library versions
    (yfinance's ``dividendYield`` has been both a fraction and a percent).
    Rather than guess and risk a silent 100x error, emit the raw number and
    say plainly that it must not be converted.
    """
    if value is None:
        return None
    field = f"`{raw_field}`" if raw_field else "this field"
    number = _as_float(value)
    shown = f"{number:g}" if number is not None else str(value)
    return (
        f"{shown}  [RAW vendor value — the unit convention for {field} is not "
        f"normalized here; do not convert, scale, or restate it as a percentage]"
    )


def fmt_text(value) -> str | None:
    """Render a non-numeric field, dropping empty strings."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


# Dispatch table used by vendor modules to attach a unit kind to each field.
FORMATTERS: dict[str, Callable] = {
    "text": lambda v, **_: fmt_text(v),
    "money": lambda v, currency="USD", **_: fmt_money(v, currency),
    "price": lambda v, currency="USD", **_: fmt_price(v, currency),
    "multiple": lambda v, **_: fmt_multiple(v),
    "fraction_pct": lambda v, raw_field=None, **_: fmt_fraction_as_pct(v, raw_field),
    "percent_and_multiple": lambda v, raw_field=None, **_: fmt_percent_with_multiple(v, raw_field),
    "plain": lambda v, **_: fmt_plain(v),
    "raw": lambda v, raw_field=None, **_: fmt_raw(v, raw_field),
}


def render_field(kind: str, value, *, raw_field: str | None = None, currency: str = "USD") -> str | None:
    """Format ``value`` according to ``kind``; None when there is nothing to show."""
    formatter = FORMATTERS.get(kind)
    if formatter is None:
        raise KeyError(f"Unknown unit kind: {kind!r}")
    return formatter(value, raw_field=raw_field, currency=currency)


def pct_change(new, old) -> float | None:
    """Percent change from ``old`` to ``new``, or None when it is undefined.

    A zero or negative base makes a percent change meaningless (a swing from
    -100 to +50 is not "+150%"), so those return None rather than a number
    that reads as growth.
    """
    new_value, old_value = _as_float(new), _as_float(old)
    if new_value is None or old_value is None or old_value <= 0:
        return None
    return (new_value / old_value - 1.0) * 100.0


def safe_ratio(numerator, denominator) -> float | None:
    """``numerator / denominator``, or None when the denominator is unusable."""
    num, den = _as_float(numerator), _as_float(denominator)
    if num is None or den is None or den == 0:
        return None
    return num / den
