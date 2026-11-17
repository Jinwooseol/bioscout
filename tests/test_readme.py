"""The README's results block and its chart: drawn from a report, and held to the committed baseline."""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pytest

from bioscout.evals import chart, cli, readme
from bioscout.evals.gate import Baseline

ROOT = Path(__file__).parent.parent
SVG = "{http://www.w3.org/2000/svg}"
NAMES = {"B": "Naive RAG", "C": "Agent"}
NUMBERS = {
    "B": {"citation_rate": 0.9, "citation_coverage": 0.6, "citation_precision": 0.9, "usd_per_query": 0.02},
    "C": {
        "citation_rate": 0.997,
        "citation_coverage": 0.93,
        "citation_precision": 0.86,
        "usd_per_query": 0.15,
    },
}


def report(numbers: dict[str, dict[str, Any]] | None = None, **meta_changes: Any) -> dict[str, Any]:
    numbers = NUMBERS if numbers is None else numbers
    rows = []
    for key, values in numbers.items():
        row = {"variant": key, "fabricated_pmid_rate": 0.0, "accuracy": None, "gold_in_context_rate": None}
        rows.append({**row, "dataset": "all", "n": 115, **values})
        rows.append({**row, "dataset": "domain", "n": 15, **values})
        rows.append(
            {**row, "dataset": "pubmedqa", "n": 100, **values, "accuracy": 0.76, "gold_in_context_rate": 0.9}
        )
    meta = {
        "run_id": "run-1",
        "model": "m",
        "effort": "medium",
        "items": 115,
        "datasets": ["pubmedqa", "domain"],
        "variants": {key: {"name": NAMES.get(key, f"Variant {key}"), "prompt_sha": "x"} for key in numbers},
        "judge": {"prompt_sha": "jjj", "sentences": 1300},
    }
    return {"meta": {**meta, **meta_changes}, "rows": rows}


def baseline(made: dict[str, Any] | None = None) -> Baseline:
    return Baseline("evals/reports/run-1.json", report() if made is None else made)


def drawn(svg: str) -> ET.Element:
    return ET.fromstring(svg)


def test_the_readme_and_its_charts_show_the_report_the_gate_checks():
    committed = Baseline.load((ROOT / "evals" / "baseline.json").read_text(encoding="utf-8"), ROOT)
    out_of_date = readme.stale(ROOT, committed)
    assert out_of_date == [], f"run `uv run bioscout-eval readme`: {out_of_date} no longer show the baseline"


def test_a_bar_splits_every_sentence_into_backed_cited_and_uncited():
    row = chart.chart_rows(report())[0]
    assert row.label == "B. Naive RAG"
    assert row.shares == pytest.approx((0.6, 0.3, 0.1))  # coverage, cited but not backed, no citation
    assert sum(row.shares) == pytest.approx(1.0)

    parts = chart.spans(row.shares)
    assert all(parts)
    starts = [part[0] for part in parts]
    ends = [part[0] + part[1] for part in parts]
    assert starts[0] == chart.BAR_X and ends[-1] == pytest.approx(chart.BAR_X + chart.BAR_W)
    # Two parts that touch are kept apart by a gap, cut half from each, so a part stays where the axis says.
    assert [starts[n + 1] - ends[n] for n in range(2)] == pytest.approx([chart.GAP, chart.GAP])
    assert ends[0] + chart.GAP / 2 == pytest.approx(chart.BAR_X + 0.6 * chart.BAR_W)


def test_a_part_too_thin_to_see_is_left_out_and_the_bar_ends_round_on_the_last_part_drawn():
    backed, unbacked, uncited = chart.spans((0.93, 0.067, 0.003))
    assert uncited is None  # a third of a percent is under a pixel
    assert backed[0] == chart.BAR_X and unbacked[0] - (backed[0] + backed[1]) == pytest.approx(chart.GAP)
    assert unbacked[0] + unbacked[1] == pytest.approx(chart.BAR_X + 0.997 * chart.BAR_W)  # no gap after it

    theme = chart.LIGHT
    svg = drawn(chart.results_svg(report({"C": NUMBERS["C"]}), theme))
    shapes = {shape.get("fill"): shape.tag for shape in svg if shape.tag in (f"{SVG}rect", f"{SVG}path")}
    assert shapes[theme.backed] == f"{SVG}rect"  # square at the baseline, and another part follows it
    assert shapes[theme.unbacked] == f"{SVG}path"  # the last part drawn carries the rounded end
    assert chart.spans((0.0, 0.0, 1.0)) == [None, None, (chart.BAR_X, chart.BAR_W)]  # nothing cited at all


def test_only_a_judged_variant_gets_a_bar_and_the_table_says_what_was_not_run():
    unjudged = {
        "citation_rate": 0.9,
        "citation_coverage": None,
        "citation_precision": None,
        "usd_per_query": 0.02,
    }
    made = report({"B": unjudged, "C": NUMBERS["C"]})
    assert [row.label for row in chart.chart_rows(made)] == ["C. Agent"]

    table = readme.table(made).splitlines()
    assert table[0].startswith("| Variant | Fabricated PMIDs | Sentences cited | Citation precision |")
    cells = {line.split(" | ")[0].removeprefix("| "): line for line in table[2:]}
    assert list(cells) == [f"{key}. {variant.name}" for key, variant in sorted(readme.VARIANTS.items())]
    assert cells["B. Naive RAG"] == "| B. Naive RAG | 0.0% | 90.0% | n/a | n/a | 90% | 76% | $0.020 |"
    assert cells["C. Agent"] == "| C. Agent | 0.0% | 99.7% | 86.0% | 93.0% | 90% | 76% | $0.150 |"
    assert cells["D. Agent + verifier"] == "| D. Agent + verifier | built, not run yet | | | | | | |"
    assert all(line.count("|") == table[0].count("|") for line in table)  # every row has every column

    with pytest.raises(ValueError):
        chart.results_svg(report({"B": unjudged}), chart.LIGHT)


def test_the_same_report_gives_the_same_bytes_and_the_two_themes_differ_only_in_colour():
    made = report()
    light, dark = (chart.results_svg(made, theme) for theme in (chart.LIGHT, chart.DARK))
    assert light == chart.results_svg(made, chart.LIGHT)
    assert light != dark

    def colourless(svg: str) -> str:
        return re.sub(r"#[0-9a-f]{6}", "#", svg)

    assert colourless(light) == colourless(dark)
    assert light.endswith("</svg>\n") and "\r" not in light


def test_the_chart_is_plain_svg_that_says_in_words_what_it_shows():
    made = report()
    made["meta"]["variants"]["C"]["name"] = 'Agent <v2> & "more"'
    theme = chart.DARK
    text = chart.results_svg(made, theme)
    svg = drawn(text)  # a name with markup in it is written as text, so the file still parses
    assert svg.get("role") == "img" and svg.find(f"{SVG}title").text
    described = svg.find(f"{SVG}desc").text
    assert 'C. Agent <v2> & "more": 93.0% backed, $0.150' in described
    assert described == chart.describe(chart.chart_rows(made))

    # Nothing in it runs or loads: no script, no link, no style sheet, no image.
    assert {child.tag.removeprefix(SVG) for child in svg.iter()} <= {
        "svg", "title", "desc", "rect", "path", "line", "text", "tspan"}  # fmt: skip
    assert "href" not in text and text.count("http") == 1  # the SVG namespace

    # Words are never in a series colour; only the legend's swatches are.
    series = {theme.backed, theme.unbacked}
    coloured = [
        el.text for el in svg.iter() if el.tag in (f"{SVG}text", f"{SVG}tspan") and el.get("fill") in series
    ]
    assert coloured == [chart.SWATCH, chart.SWATCH]
    words = "".join(svg.itertext())
    assert "93.0% backed" in words and "$0.150" in words and "Cost per question" in words
    assert all(name in words for _, name in chart.PARTS)


def test_the_note_under_the_chart_says_what_the_numbers_rest_on():
    first, second = chart.notes(report())
    assert first == "115 questions (100 PubMedQA, 15 domain), answered by every variant with m."
    assert second.endswith("an LLM judge that has not been checked against hand labels yet.")

    checked = report(calibration={"pairs": 60, "agreement": 0.9, "kappa": 0.8123})
    assert chart.notes(checked)[1].endswith(
        "an LLM judge; on 60 hand-labelled pairs it agrees 90% of the time (Cohen's kappa 0.81)."
    )


def test_the_block_names_its_report_and_replaces_only_what_is_between_the_markers():
    block = readme.block(baseline())
    assert block.startswith(readme.BEGIN) and block.endswith(readme.END)
    assert "[`evals/reports/run-1.md`](evals/reports/run-1.md)" in block
    assert "115 questions, model `m`, effort `medium`, 1,300 cited sentences judged." in block
    assert f'srcset="{readme.CHARTS["dark"]}"' in block and f'src="{readme.CHARTS["light"]}"' in block
    alt = re.search(r'alt="([^"]*)"', block).group(1)
    assert "B. Naive RAG: 60.0% backed, $0.020" in alt

    page = f"# Title\n\nBefore.\n\n{readme.BEGIN}\nold numbers\n{readme.END}\n\nAfter.\n"
    new = readme.replace_block(page, block)
    assert new == f"# Title\n\nBefore.\n\n{block}\n\nAfter.\n"
    assert readme.replace_block(new, block) == new  # writing twice changes nothing

    for broken in (
        "no markers",
        f"{readme.END}\n{readme.BEGIN}",
        f"{readme.BEGIN}{readme.BEGIN}{readme.END}",
    ):
        with pytest.raises(ValueError):
            readme.replace_block(broken, block)


def test_the_command_writes_what_is_out_of_date_and_its_check_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    (tmp_path / "evals" / "reports").mkdir(parents=True)
    (tmp_path / "evals" / "reports" / "run-1.json").write_text(json.dumps(report()), encoding="utf-8")
    (tmp_path / "evals" / "baseline.json").write_text(
        json.dumps({"report": "evals/reports/run-1.json"}), encoding="utf-8"
    )
    page = f"Before.\n\n{readme.BEGIN}\n{readme.END}\n\nAfter.\n"
    (tmp_path / "README.md").write_text(page, encoding="utf-8", newline="\n")
    monkeypatch.chdir(tmp_path)
    charts = sorted(readme.CHARTS.values())

    def run(check: bool) -> tuple[int, str]:
        code = cli.cmd_readme(argparse.Namespace(baseline=None, evals_dir=Path("evals"), check=check))
        return code, capsys.readouterr().out

    code, said = run(check=True)
    assert code == 1 and [line for line in said.splitlines() if line.startswith("out of date")] == [
        "out of date: README.md", *(f"out of date: {name}" for name in charts)]  # fmt: skip
    assert (tmp_path / "README.md").read_text(encoding="utf-8") == page  # the check wrote nothing
    assert not (tmp_path / "docs").exists()

    code, said = run(check=False)
    assert code == 0 and said.splitlines() == ["wrote README.md", *(f"wrote {name}" for name in charts)]
    written = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert (
        written.startswith("Before.\n\n") and written.endswith("\n\nAfter.\n") and "| C. Agent |" in written
    )
    assert b"\r" not in (tmp_path / charts[0]).read_bytes()
    assert run(check=True)[0] == 0 and run(check=False) == (
        0, "README.md and its charts already show evals/reports/run-1.json\n")  # fmt: skip

    # The baseline moves to a report with other numbers: the page and the charts are out of date again.
    moved = report({"C": {**NUMBERS["C"], "citation_coverage": 0.5}})
    (tmp_path / "evals" / "reports" / "run-1.json").write_text(json.dumps(moved), encoding="utf-8")
    assert run(check=True)[0] == 1
