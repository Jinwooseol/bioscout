"""The results block of README.md: the chart and the table, written from the baseline report.

A README shows numbers, and numbers typed by hand drift from the evidence. The block between the two
markers is therefore generated from the report named in evals/baseline.json, which is the report the
eval gate checks, and so are the two chart files it shows (one for a light page, one for a dark one).
A test fails when the README or a chart differs from what that report gives.

  uv run bioscout-eval readme          rewrite the block and the charts
  uv run bioscout-eval readme --check  only compare; exit 1 when one is out of date
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from bioscout.evals import chart
from bioscout.evals.gate import Baseline
from bioscout.evals.variants import VARIANTS

README = "README.md"
BEGIN = "<!-- results:begin -->"
END = "<!-- results:end -->"
CHARTS = {"light": "docs/assets/results-light.svg", "dark": "docs/assets/results-dark.svg"}
NOT_RUN = "built, not run yet"
MISSING = "n/a"

# (key in a report row, heading, the dataset the number is taken from, how it is written)
COLUMNS = [
    ("fabricated_pmid_rate", "Fabricated PMIDs", "all", "pct"),
    ("citation_rate", "Sentences cited", "all", "pct"),
    ("citation_precision", "Citation precision", "all", "pct"),
    ("citation_coverage", "Citation coverage", "all", "pct"),
    ("gold_in_context_rate", "Source paper used (PubMedQA)", "pubmedqa", "answers"),
    ("accuracy", "Accuracy (PubMedQA)", "pubmedqa", "answers"),
    ("usd_per_query", "Cost / question", "all", "usd"),
]


def cell(value: float | None, kind: str) -> str:
    if value is None:
        return MISSING
    if kind == "usd":
        return f"${value:.3f}"
    if kind == "answers":  # one answer per question: a tenth of a point would be precision it lacks
        return f"{value * 100:.0f}%"
    return f"{value * 100:.1f}%"


def table(report: dict[str, Any]) -> str:
    """One row per variant the code has: its numbers, or a note that it has not been measured."""
    overall = chart.overall(report)
    by_dataset = {(row["variant"], row["dataset"]): row for row in report["rows"]}
    lines = [
        "| Variant | " + " | ".join(heading for _, heading, _, _ in COLUMNS) + " |",
        "|" + " --- |" * (len(COLUMNS) + 1),
    ]
    for key in sorted({*VARIANTS, *overall}):
        if key not in overall:
            lines.append(f"| {key}. {VARIANTS[key].name} | {NOT_RUN} |" + " |" * (len(COLUMNS) - 1))
            continue
        cells = []
        for name, _, dataset, kind in COLUMNS:
            row = overall[key] if dataset == "all" else by_dataset.get((key, dataset), {})
            cells.append(cell(row.get(name), kind))
        lines.append(f"| {chart.label_of(key, report)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def block(baseline: Baseline) -> str:
    """Everything between the markers, markers included."""
    report = baseline.report
    meta = report["meta"]
    shown = Path(baseline.report_path).with_suffix(".md").as_posix()
    alt = chart.describe(chart.chart_rows(report)).replace('"', "'")
    source = f"{meta['items']} questions, model `{meta['model']}`, effort `{meta['effort']}`"
    if judge := meta.get("judge"):
        source += f", {judge['sentences']:,} cited sentences judged"
    return "\n".join(
        [
            BEGIN,
            "<!-- Written by `bioscout-eval readme` from the report named in evals/baseline.json. "
            "Change the report, not these lines. -->",
            "",
            "<picture>",
            f'  <source media="(prefers-color-scheme: dark)" srcset="{CHARTS["dark"]}">',
            f'  <img src="{CHARTS["light"]}" width="{chart.WIDTH}" alt="{alt}">',
            "</picture>",
            "",
            table(report),
            "",
            f"Chart and table are written from [`{shown}`]({shown}), the report the eval gate checks: "
            f"{source}.",
            END,
        ]
    )


def replace_block(readme: str, new: str) -> str:
    if readme.count(BEGIN) != 1 or readme.count(END) != 1 or readme.index(BEGIN) > readme.index(END):
        raise ValueError(f"{README} needs one {BEGIN} followed by one {END}")
    return readme[: readme.index(BEGIN)] + new + readme[readme.index(END) + len(END) :]


def build(root: Path, baseline: Baseline) -> dict[str, str]:
    """What each generated file should hold, keyed by its path under root."""
    files = {path: chart.results_svg(baseline.report, chart.THEMES[name]) for name, path in CHARTS.items()}
    files[README] = replace_block((root / README).read_text(encoding="utf-8"), block(baseline))
    return files


def _differing(root: Path, wanted: dict[str, str]) -> list[str]:
    def same(path: str) -> bool:
        target = root / path
        return target.exists() and target.read_text(encoding="utf-8") == wanted[path]

    return sorted(path for path in wanted if not same(path))


def stale(root: Path, baseline: Baseline) -> list[str]:
    """The generated files that are missing or differ from what the baseline report gives."""
    return _differing(root, build(root, baseline))


def write(root: Path, baseline: Baseline) -> list[str]:
    """Bring the generated files up to date and return the ones that changed."""
    wanted = build(root, baseline)
    changed = _differing(root, wanted)
    for path in changed:
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(wanted[path], encoding="utf-8", newline="\n")
    return changed
