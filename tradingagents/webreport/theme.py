"""Shared CSS and page shell for the self-contained HTML reports.

Every page this package emits is a single file with no external requests: no
CDN, no webfont, no image host. That is deliberate — reports are opened from
``file://``, emailed as attachments, and archived beside the run they describe,
none of which can rely on a network.

The colour values are not chosen by eye. They come from the data-visualisation
palette and were checked with its validator; the per-role notes below record
what passed and what carries an obligation.
"""

from __future__ import annotations

# Chart chrome and ink, straight from the reference palette.
LIGHT_TOKENS = {
    "surface": "#fcfcfb",
    "plane": "#f9f9f7",
    "text-primary": "#0b0b0b",
    "text-secondary": "#52514e",
    "text-muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "border": "rgba(11,11,11,0.10)",
    "positive": "#184f95",
    "negative": "#8f2020",
}

DARK_TOKENS = {
    "surface": "#1a1a19",
    "plane": "#0d0d0d",
    "text-primary": "#ffffff",
    "text-secondary": "#c3c2b7",
    "text-muted": "#898781",
    "grid": "#2c2c2a",
    "axis": "#383835",
    "border": "rgba(255,255,255,0.10)",
    "positive": "#b7d3f6",
    "negative": "#f7b3b3",
}

# Categorical hues for strategy/arm series, in the palette's fixed order. Never
# cycled: a fifth series folds into "other" rather than reusing slot 1.
# Validator (adjacent pairlist, the right one for lines and bars): light worst
# CVD dE 9.1 / normal 22.9; dark worst CVD 8.4 / normal 19.8 — all pass.
SERIES_LIGHT = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
SERIES_DARK = ("#3987e5", "#d95926", "#199e70", "#c98500")

# The five-tier rating scale is ordered and signed, so it takes a diverging
# ramp — two one-hue arms around a neutral midpoint — not categorical hues.
# Each arm validates as an ordinal ramp on its own; across the whole scale the
# checks that govern a diverging scale pass in both modes (light worst CVD dE
# 8.7 / normal 15.3; dark worst CVD 15.9 / normal 17.9).
#
# The validator's "lightness band" and "chroma floor" checks do fail here, and
# that is expected rather than ignored: both are scoped to categorical palettes,
# and a diverging ramp is required to have a low-chroma neutral midpoint and
# outer steps beyond the categorical band. Those two properties are what make it
# diverging.
#
# The neutral midpoint sits below 3:1 against the surface in both modes, which
# obligates relief: every chart drawn from this ramp ships visible segment
# labels and is accompanied by the equivalent data table.
RATING_SCALE_LIGHT = {
    "Buy": "#184f95",
    "Overweight": "#5598e7",
    "Hold": "#898781",
    "Underweight": "#e34948",
    "Sell": "#8f2020",
}
RATING_SCALE_DARK = {
    "Buy": "#b7d3f6",
    "Overweight": "#3987e5",
    "Hold": "#504f4a",
    "Underweight": "#e66767",
    "Sell": "#f7b3b3",
}


def _token_block(tokens: dict, series: tuple[str, ...], ratings: dict) -> str:
    lines = [f"  --{name}: {value};" for name, value in tokens.items()]
    lines += [f"  --series-{i + 1}: {hex_};" for i, hex_ in enumerate(series)]
    lines += [
        f"  --rating-{name.lower()}: {hex_};" for name, hex_ in ratings.items()
    ]
    return "\n".join(lines)


def build_css() -> str:
    """Full stylesheet, including both themes.

    Dark values are declared under the OS media query *and* the ``data-theme``
    attribute, so an explicit toggle beats the OS setting in both directions.
    The ``:not()`` guard lets a light stamp win over OS-dark; ``:where()`` keeps
    the media block's specificity below the toggle's.
    """
    light = _token_block(LIGHT_TOKENS, SERIES_LIGHT, RATING_SCALE_LIGHT)
    dark = _token_block(DARK_TOKENS, SERIES_DARK, RATING_SCALE_DARK)
    return f"""
:root {{
  color-scheme: light;
{light}
  --radius: 10px;
  --font: system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}}
@media (prefers-color-scheme: dark) {{
  :root:where(:not([data-theme="light"])) {{
    color-scheme: dark;
{dark}
  }}
}}
:root[data-theme="dark"] {{
  color-scheme: dark;
{dark}
}}

* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  padding: 0 1.25rem 5rem;
  background: var(--plane);
  color: var(--text-primary);
  font-family: var(--font);
  font-size: 15px;
  line-height: 1.65;
  -webkit-text-size-adjust: 100%;
}}
.wrap {{ max-width: 60rem; margin: 0 auto; }}

header.page {{
  padding: 2.5rem 0 1.25rem;
  border-bottom: 1px solid var(--border);
  margin-bottom: 2rem;
}}
header.page h1 {{ margin: 0 0 .35rem; font-size: 1.7rem; letter-spacing: -0.015em; }}
header.page .sub {{ color: var(--text-secondary); font-size: .92rem; }}

h2 {{
  margin: 2.75rem 0 .9rem;
  font-size: 1.22rem;
  letter-spacing: -0.01em;
  padding-bottom: .4rem;
  border-bottom: 1px solid var(--border);
}}
h3 {{ margin: 1.9rem 0 .6rem; font-size: 1.02rem; }}
h4 {{ margin: 1.4rem 0 .5rem; font-size: .95rem; color: var(--text-secondary); }}
p {{ margin: .7rem 0; }}
a {{ color: var(--series-1); }}

.card {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.25rem 1.4rem;
  margin: 1.25rem 0;
}}
figure {{ margin: 0; }}
figcaption {{
  color: var(--text-secondary);
  font-size: .86rem;
  margin-top: .7rem;
}}

/* Wide content scrolls inside its own box; the page never scrolls sideways. */
.scroll-x {{ overflow-x: auto; -webkit-overflow-scrolling: touch; }}

table {{ border-collapse: collapse; width: 100%; font-size: .88rem; }}
th, td {{
  text-align: left;
  padding: .5rem .7rem;
  border-bottom: 1px solid var(--border);
  white-space: nowrap;
}}
th {{
  color: var(--text-secondary);
  font-weight: 600;
  font-size: .78rem;
  text-transform: uppercase;
  letter-spacing: .04em;
}}
tbody tr:last-child td {{ border-bottom: none; }}
td.num, th.num {{ text-align: right; font-variant-numeric: tabular-nums; }}

code {{
  font-family: var(--mono);
  font-size: .87em;
  background: var(--plane);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: .08em .34em;
}}
pre {{
  background: var(--plane);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: .9rem 1rem;
  overflow-x: auto;
}}
pre code {{ background: none; border: none; padding: 0; }}
blockquote {{
  margin: 1rem 0;
  padding: .1rem 1rem;
  border-left: 3px solid var(--axis);
  color: var(--text-secondary);
}}

/* Chart primitives ------------------------------------------------------ */
svg.chart {{ display: block; width: 100%; height: auto; }}
svg.chart text {{ font-family: var(--font); fill: var(--text-secondary); }}
svg.chart .tick {{ font-size: 11px; fill: var(--text-muted); }}
svg.chart .row-label {{ font-size: 12.5px; fill: var(--text-primary); }}
svg.chart .value-label {{ font-size: 12px; fill: var(--text-primary); font-variant-numeric: tabular-nums; }}
svg.chart .grid-line {{ stroke: var(--grid); stroke-width: 1; }}
svg.chart .zero-line {{ stroke: var(--axis); stroke-width: 1.5; }}
svg.chart .mark:hover {{ opacity: .78; }}
svg.chart .series-line {{ fill: none; stroke-width: 2; }}

.legend {{
  display: flex;
  flex-wrap: wrap;
  gap: .35rem 1.1rem;
  margin: .8rem 0 0;
  padding: 0;
  list-style: none;
  font-size: .84rem;
  color: var(--text-secondary);
}}
.legend li {{ display: flex; align-items: center; gap: .4rem; }}
.legend .swatch {{
  width: 11px; height: 11px; border-radius: 3px; flex: none;
  box-shadow: 0 0 0 1px var(--border) inset;
}}

.note {{
  font-size: .88rem;
  color: var(--text-secondary);
  border-left: 3px solid var(--axis);
  padding: .1rem .9rem;
  margin: 1rem 0;
}}
.verdict {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-left: 3px solid var(--series-1);
  border-radius: var(--radius);
  padding: .9rem 1.1rem;
  margin: 1.1rem 0;
}}

/* Section nav for the analysis report ----------------------------------- */
nav.toc {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1rem 1.2rem;
  margin-bottom: 2rem;
}}
nav.toc ul {{ margin: .3rem 0 0; padding-left: 1.1rem; list-style: none; }}
nav.toc li::before {{ content: ""; color: var(--text-muted); margin-right: .5rem; }}
nav.toc li {{ margin: .15rem 0; }}
nav.toc a {{ text-decoration: none; }}
nav.toc a:hover {{ text-decoration: underline; }}

.theme-toggle {{
  position: fixed; top: .9rem; right: .9rem;
  background: var(--surface);
  color: var(--text-secondary);
  border: 1px solid var(--border);
  border-radius: 999px;
  padding: .35rem .8rem;
  font: inherit; font-size: .8rem;
  cursor: pointer;
}}
.theme-toggle:hover {{ color: var(--text-primary); }}

@media print {{
  .theme-toggle {{ display: none; }}
  body {{ background: #fff; }}
}}
""".strip()


# Tiny, and the only script on the page. Reports are archived and re-opened
# long after the run, so the toggle has to work from a bare file:// load with
# no build step and no storage assumptions beyond localStorage.
THEME_TOGGLE_JS = """
(function () {
  var root = document.documentElement;
  var saved = null;
  try { saved = localStorage.getItem('ta-theme'); } catch (e) {}
  if (saved) root.setAttribute('data-theme', saved);
  var btn = document.querySelector('.theme-toggle');
  if (!btn) return;
  btn.addEventListener('click', function () {
    var dark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    var current = root.getAttribute('data-theme') || (dark ? 'dark' : 'light');
    var next = current === 'dark' ? 'light' : 'dark';
    root.setAttribute('data-theme', next);
    try { localStorage.setItem('ta-theme', next); } catch (e) {}
  });
})();
""".strip()


def render_page(title: str, body: str, *, subtitle: str = "") -> str:
    """Wrap ``body`` in a complete, self-contained HTML document."""
    sub = f'<div class="sub">{subtitle}</div>' if subtitle else ""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
{build_css()}
</style>
</head>
<body>
<button class="theme-toggle" type="button" aria-label="Toggle light and dark theme">theme</button>
<div class="wrap">
<header class="page">
<h1>{title}</h1>
{sub}
</header>
{body}
</div>
<script>
{THEME_TOGGLE_JS}
</script>
</body>
</html>
"""
