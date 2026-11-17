"""The results chart of the README, drawn as SVG from the rows of a report.

One figure, two panels that share their rows, one row per judged variant:

  left   what the sentences of the answers rest on. One bar per variant, split three ways: backed by
         the abstracts the sentence cites (citation coverage), cited but not fully backed, no citation.
  right  what one question costs.

The SVG is written out by hand rather than with a plotting library, so one report always gives the same
bytes and a test can hold the committed chart to the committed report. It carries its own surface, so
it reads the same on any page. The two series colours were checked for colour-blind separation and for
contrast against each surface; text is never in a series colour, and the split a reader cannot see in
the bar (a part under a pixel wide) is still in the table that sits under the chart.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from xml.sax.saxutils import escape

DATASET_NAMES = {"pubmedqa": "PubMedQA", "domain": "domain"}

# Geometry, in the units of the viewBox (one unit is a pixel at full size).
WIDTH = 860
PAD = 28
BAR_X = 220  # the shares start here, right of the column of variant names
BAR_W = 320
VALUE_X = BAR_X + BAR_W + 10
COST_X = 672
COST_W = 110
ROW_TOP = 86
ROW_H = 34
BAR_H = 18
GAP = 2  # of surface between two parts of a bar
RADIUS = 4  # of the end a bar grows towards; its start is square
MIN_PART = 3  # a part narrower than this would vanish between its gaps, so it is not drawn
SWATCH = "■"  # a filled square, drawn in a part's colour in front of its name
FONT = "system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"


@dataclass(frozen=True)
class Theme:
    name: str
    surface: str
    edge: str  # the hairline around the figure, drawn at 10% opacity
    ink: str
    ink_secondary: str
    muted: str  # axis labels, the cost bars and, at 45% opacity, the sentences with no citation
    grid: str
    axis: str
    backed: str
    unbacked: str


LIGHT = Theme("light", "#fcfcfb", "#0b0b0b", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7",
              backed="#2a78d6", unbacked="#eb6834")  # fmt: skip
DARK = Theme("dark", "#1a1a19", "#ffffff", "#f0efec", "#c3c2b7", "#898781", "#2c2c2a", "#383835",
             backed="#3987e5", unbacked="#d95926")  # fmt: skip
THEMES = {theme.name: theme for theme in (LIGHT, DARK)}

PARTS = (
    ("backed", "Backed by the abstracts it cites"),
    ("unbacked", "Cited, not fully backed"),
    ("uncited", "No citation"),
)


@dataclass(frozen=True)
class Row:
    """One variant: how its answer sentences split (three shares that add up to 1) and what it costs."""

    label: str
    backed: float
    unbacked: float
    uncited: float
    usd: float

    @property
    def shares(self) -> tuple[float, float, float]:
        return (self.backed, self.unbacked, self.uncited)


def overall(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The row of each variant over every question: its 'all' row, or its only row in a one-dataset run."""
    rows: dict[str, dict[str, Any]] = {}
    for row in report["rows"]:
        if row["dataset"] == "all" or row["variant"] not in rows:
            rows[row["variant"]] = row
    return dict(sorted(rows.items()))


def label_of(key: str, report: dict[str, Any]) -> str:
    return f"{key}. {report['meta']['variants'][key]['name']}"


def chart_rows(report: dict[str, Any]) -> list[Row]:
    """The variants that can be drawn: judged, with a cost. Coverage counts every sentence of the answers,
    cited or not, so the three shares are parts of one whole."""
    rows = []
    for key, row in overall(report).items():
        cited, backed, usd = row.get("citation_rate"), row.get("citation_coverage"), row.get("usd_per_query")
        if cited is None or backed is None or usd is None:
            continue
        rows.append(Row(label_of(key, report), backed, max(cited - backed, 0.0), max(1.0 - cited, 0.0), usd))
    return rows


def describe(rows: list[Row]) -> str:
    """The chart in words, for readers who do not see it."""
    each = "; ".join(f"{row.label}: {row.backed:.1%} backed, ${row.usd:.3f}" for row in rows)
    return (
        "Share of answer sentences backed by the abstracts they cite, and cost per question, "
        f"by variant. {each}."
    )


def notes(report: dict[str, Any]) -> list[str]:
    """What the numbers rest on: the questions, the model, and how far the judge can be trusted."""
    meta = report["meta"]
    counted = {row["dataset"]: row["n"] for row in report["rows"] if row["dataset"] != "all"}
    sizes = {name: counted[name] for name in meta["datasets"] if name in counted}
    split = ", ".join(f"{n} {DATASET_NAMES.get(name, name)}" for name, n in sizes.items())
    first = f"{sum(sizes.values())} questions ({split}), answered by every variant with {meta['model']}."
    second = "Whether an abstract backs a sentence is decided by an LLM judge"
    if calibration := meta.get("calibration"):
        second += (
            f"; on {calibration['pairs']} hand-labelled pairs it agrees "
            f"{calibration['agreement']:.0%} of the time (Cohen's kappa {calibration['kappa']:.2f})."
        )
    else:
        second += " that has not been checked against hand labels yet."
    return [first, second]


def num(value: float) -> str:
    """A coordinate: two decimals at most, no trailing zeros, never '-0'."""
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def bar(x: float, y: float, width: float, fill: str, *, rounded: bool, opacity: float | None = None) -> str:
    """A bar that grows to the right from x: square where it starts, rounded where it ends."""
    look = f'fill="{fill}"' + (f' fill-opacity="{num(opacity)}"' if opacity is not None else "")
    r = min(RADIUS, width, BAR_H / 2) if rounded else 0
    if r <= 0:
        return f'<rect x="{num(x)}" y="{num(y)}" width="{num(width)}" height="{BAR_H}" {look}/>'
    path = (
        f"M{num(x)} {num(y)}h{num(width - r)}a{num(r)} {num(r)} 0 0 1 {num(r)} {num(r)}"
        f"v{num(BAR_H - 2 * r)}a{num(r)} {num(r)} 0 0 1 {num(-r)} {num(r)}h{num(r - width)}z"
    )
    return f'<path d="{path}" {look}/>'


def spans(shares: tuple[float, ...]) -> list[tuple[float, float] | None]:
    """Where each part of a split bar starts and how wide it is drawn; None for a part too thin to draw.

    A part keeps its place on the axis, so the grid reads true; the gap between two drawn parts is cut
    half from each."""
    edges = [BAR_X]
    for share in shares:
        edges.append(edges[-1] + share * BAR_W)
    drawn = [edges[i + 1] - edges[i] >= MIN_PART for i in range(len(shares))]
    out: list[tuple[float, float] | None] = []
    for i in range(len(shares)):
        if not drawn[i]:
            out.append(None)
            continue
        left = edges[i] + (GAP / 2 if any(drawn[:i]) else 0)
        right = edges[i + 1] - (GAP / 2 if any(drawn[i + 1 :]) else 0)
        out.append((left, right - left))
    return out


def text(x: float, y: float, body: str, size: int, fill: str, **attrs: str) -> str:
    extra = "".join(f' {name.replace("_", "-")}="{value}"' for name, value in attrs.items())
    return f'<text x="{num(x)}" y="{num(y)}" font-size="{size}" fill="{fill}"{extra}>{body}</text>'


def results_svg(report: dict[str, Any], theme: Theme) -> str:
    rows = chart_rows(report)
    if not rows:
        raise ValueError("the report has no judged variant to draw")
    fills = {
        "backed": (theme.backed, None),
        "unbacked": (theme.unbacked, None),
        "uncited": (theme.muted, 0.45),
    }
    bottom = ROW_TOP + ROW_H * len(rows)
    lines = notes(report)
    height = bottom + 62 + 17 * len(lines)
    summary = describe(rows)
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {height}" width="{WIDTH}" '
        f'height="{height}" role="img" aria-labelledby="title desc" font-family="{FONT}">',
        '<title id="title">BioScout evaluation results by variant</title>',
        f'<desc id="desc">{escape(summary)}</desc>',
        f'<rect x="0.5" y="0.5" width="{WIDTH - 1}" height="{height - 1}" rx="8" fill="{theme.surface}" '
        f'stroke="{theme.edge}" stroke-opacity="0.1"/>',
        text(PAD, 42, "What the sentences of the answers rest on", 14, theme.ink, font_weight="600"),
        text(COST_X, 42, "Cost per question", 14, theme.ink, font_weight="600"),
    ]

    # The legend, in the order of the parts along a bar. It is one line of text whose swatches are
    # glyphs, so the names follow each other evenly in whatever font the reader has: no width is guessed.
    legend = []
    for n, (key, name) in enumerate(PARTS):
        fill, opacity = fills[key]
        look = f'fill="{fill}"' + (f' fill-opacity="{num(opacity)}"' if opacity is not None else "")
        gap = ' dx="20"' if n else ""
        legend.append(
            f'<tspan{gap} font-size="14" {look}>{SWATCH}</tspan><tspan dx="6">{escape(name)}</tspan>'
        )
    out.append(text(PAD, 66, "".join(legend), 12, theme.ink_secondary))

    for tick in range(5):  # 0, 25, 50, 75 and 100% of the sentences
        at = BAR_X + BAR_W * tick / 4
        colour = theme.axis if tick == 0 else theme.grid
        out.append(
            f'<line x1="{num(at)}" y1="{ROW_TOP - 4}" x2="{num(at)}" y2="{bottom + 4}" stroke="{colour}"/>'
        )
        label = "0" if tick == 0 else f"{tick * 25}%"
        out.append(text(at, bottom + 20, label, 11, theme.muted, text_anchor="middle"))
    out.append(
        f'<line x1="{COST_X}" y1="{ROW_TOP - 4}" x2="{COST_X}" y2="{bottom + 4}" stroke="{theme.axis}"/>'
    )

    dearest = max(row.usd for row in rows)
    for i, row in enumerate(rows):
        top = ROW_TOP + ROW_H * i
        y = top + (ROW_H - BAR_H) / 2
        baseline = top + ROW_H / 2 + 4.5
        out.append(text(PAD, baseline, escape(row.label), 13, theme.ink))
        parts = spans(row.shares)
        last = max((n for n, part in enumerate(parts) if part), default=None)
        for n, ((key, _), part) in enumerate(zip(PARTS, parts, strict=True)):
            if part:
                fill, opacity = fills[key]
                out.append(bar(part[0], y, part[1], fill, rounded=n == last, opacity=opacity))
        value = (
            f'<tspan font-weight="600">{row.backed:.1%}</tspan>'
            f'<tspan fill="{theme.ink_secondary}"> backed</tspan>'
        )
        out.append(text(VALUE_X, baseline, value, 13, theme.ink))
        width = max(COST_W * row.usd / dearest, 1) if dearest else 1
        out.append(bar(COST_X, y, width, theme.muted, rounded=True))
        out.append(text(COST_X + width + 8, baseline, f"${row.usd:.3f}", 13, theme.ink, font_weight="600"))

    for i, line in enumerate(lines):
        out.append(text(PAD, bottom + 50 + 17 * i, escape(line), 12, theme.ink_secondary))
    out.append("</svg>")
    return "\n".join(out) + "\n"
