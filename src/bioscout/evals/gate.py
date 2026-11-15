"""The eval gate: the committed numbers must come from the code as it is, and must not have got worse.

CI cannot run the agent: it has no API key, and the response cache that would let it replay a run holds
PubMed abstracts, which are never committed. What CI can check, offline, is the evidence that is
committed. evals/baseline.json names the report that stands behind the current code, and the gate checks

  fresh       the report covers the whole datasets and was made with the prompts, schemas, settings,
              model, judge and questions the code has now. A change to the agent fails here until the
              agent is measured again and the baseline points at the new report.
  floor       no fabricated PMID and no failed question in the report.
  regression  no gated metric fell by more than its limit since the previous baseline.

A drop that was accepted on purpose is listed in baseline.json under "accepted", keyed
"variant/dataset/metric", with the reason as its value.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from bioscout.config import Settings
from bioscout.evals.datasets import DATASET_FILES, EvalItem, datasets_sha, load_dataset
from bioscout.evals.variants import VARIANTS
from bioscout.verifier import JUDGE_SHA

# The largest drop allowed since the previous baseline, as a share (0.03 is three points). The judge
# metrics rest on hundreds of citations; accuracy and source-in-context rest on one answer per question,
# so they move more from run to run.
GATED: dict[str, float] = {
    "citation_precision": 0.03,
    "citation_coverage": 0.03,
    "citation_rate": 0.03,
    "accuracy": 0.05,
    "gold_in_context_rate": 0.05,
}
MIN_N = 50  # in a smaller group one question is worth several points, so its drops are not gated

Status = Literal["pass", "fail", "note", "accepted"]


@dataclass(frozen=True)
class Check:
    kind: str  # fresh, floor or regression
    status: Status
    text: str


@dataclass(frozen=True)
class Baseline:
    """evals/baseline.json and the report it names."""

    report_path: str
    report: dict[str, Any]
    accepted: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, text: str, root: Path) -> Baseline:
        data = json.loads(text)
        report = json.loads((root / data["report"]).read_text(encoding="utf-8"))
        return cls(data["report"], report, dict(data.get("accepted", {})))


@dataclass(frozen=True)
class Code:
    """What a run started from the code as it is would use."""

    prompts: dict[str, str]  # variant -> prompt hash
    model: str
    effort: str
    judge_sha: str
    datasets: dict[str, list[EvalItem]]

    @classmethod
    def current(cls, datasets_dir: Path) -> Code:
        defaults = Settings()
        return cls(
            prompts={key: variant.prompt_sha for key, variant in VARIANTS.items()},
            model=defaults.llm_model,
            effort=defaults.llm_effort,
            judge_sha=JUDGE_SHA,
            datasets={name: load_dataset(name, datasets_dir) for name in DATASET_FILES},
        )


def check_fresh(report: dict[str, Any], code: Code) -> list[Check]:
    """Was the report made with what the code has now?"""
    meta = report["meta"]
    checks: list[Check] = []

    def add(ok: bool, good: str, bad: str) -> None:
        checks.append(Check("fresh", "pass" if ok else "fail", good if ok else bad))

    limit = meta.get("limit") or 0
    add(
        limit == 0,
        "the report covers the whole datasets",
        f"the report covers only the first {limit} questions of each dataset, so it cannot be the baseline",
    )
    for key, recorded in meta["variants"].items():
        sha, now = recorded["prompt_sha"], code.prompts.get(key)
        add(
            now == sha,
            f"variant {key} was measured with the prompts and settings the code has now ({sha})",
            f"variant {key} changed since it was measured ({sha} in the report, {now} in the code): "
            "run the eval again and point the baseline at the new report",
        )
    for key in code.prompts:
        if key not in meta["variants"]:
            checks.append(Check("fresh", "note", f"variant {key} has not been measured, so it is not gated"))
    add(
        (meta.get("model"), meta.get("effort")) == (code.model, code.effort),
        f"measured with the model and effort the code uses ({code.model}, {code.effort})",
        f"measured with {meta.get('model')} ({meta.get('effort')}), "
        f"but the code now uses {code.model} ({code.effort})",
    )
    judge = meta.get("judge")
    if judge:
        add(
            judge["prompt_sha"] == code.judge_sha,
            f"citations were judged with the judge prompt the code has now ({code.judge_sha})",
            f"the judge prompt changed since the citations were judged ({judge['prompt_sha']} in the "
            f"report, {code.judge_sha} in the code): judge the run again",
        )
    else:
        checks.append(Check("fresh", "note", "the report has no judge metrics, so those are not gated"))

    items = [item for name in meta["datasets"] for item in code.datasets.get(name, [])]
    if meta.get("datasets_sha"):
        add(
            datasets_sha(items) == meta["datasets_sha"],
            "the questions and labels are the ones that were measured",
            "the datasets changed since they were measured: run the eval again",
        )
    else:
        add(
            len(items) == meta.get("items"),
            f"the datasets have as many questions as were measured ({len(items)}); the report was made "
            "before dataset hashes were recorded, so their content is not compared",
            f"the datasets have {len(items)} questions but {meta.get('items')} were measured: "
            "run the eval again",
        )
    return checks


def check_floors(report: dict[str, Any]) -> list[Check]:
    """Nothing fabricated and nothing failed, per variant."""
    checks: list[Check] = []
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for row in report["rows"]:
        by_variant.setdefault(row["variant"], []).append(row)
    for variant, rows in by_variant.items():
        groups = [row for row in rows if row["dataset"] != "all"] or rows
        answers = sum(row["n"] for row in groups)
        errors = sum(row.get("errors") or 0 for row in groups)
        fabricated = [row["dataset"] for row in groups if row.get("fabricated_pmid_rate")]
        problems = []
        if errors:
            problems.append(f"{errors} question(s) failed")
        if fabricated:
            problems.append(f"fabricated PMIDs in {', '.join(fabricated)}")
        if problems:
            checks.append(Check("floor", "fail", f"variant {variant}: {'; '.join(problems)}"))
        else:
            text = f"variant {variant}: no fabricated PMID and no failed question in {answers} answers"
            checks.append(Check("floor", "pass", text))
    return checks


def check_regression(
    report: dict[str, Any], previous: dict[str, Any], accepted: dict[str, str]
) -> list[Check]:
    """No gated metric fell by more than its limit since the previous baseline."""
    checks: list[Check] = []
    before = {(row["variant"], row["dataset"]): row for row in previous["rows"]}
    held = 0
    too_small: set[str] = set()
    for row in report["rows"]:
        old = before.get((row["variant"], row["dataset"]))
        if old is None:
            continue
        if min(row["n"], old["n"]) < MIN_N:
            too_small.add(f"{row['dataset']} ({min(row['n'], old['n'])} questions)")
            continue
        for metric, limit in GATED.items():
            was, now = old.get(metric), row.get(metric)
            if was is None or now is None:
                continue
            name = f"{row['variant']}/{row['dataset']}/{metric}"
            moved = f"{name}: {was:.1%} -> {now:.1%}"
            if was - now <= limit + 1e-9:
                held += 1
            elif name in accepted:
                checks.append(Check("regression", "accepted", f"{moved} (accepted: {accepted[name]})"))
            else:
                points = f"{(was - now) * 100:.1f} points (limit {limit * 100:.0f})"
                checks.append(Check("regression", "fail", f"{moved}, a drop of {points}"))
    if held:
        text = f"{held} gated metric(s) did not fall beyond their limits since {previous['meta']['run_id']}"
        checks.insert(0, Check("regression", "pass", text))
    for group in sorted(too_small):
        checks.append(Check("regression", "note", f"{group} is too small a group to gate on drops"))
    return checks


def run_gate(baseline: Baseline, code: Code, previous: Baseline | None = None) -> list[Check]:
    checks = check_fresh(baseline.report, code) + check_floors(baseline.report)
    if previous is None:
        checks.append(Check("regression", "note", "there is no earlier baseline to compare with"))
    elif previous.report_path == baseline.report_path:
        checks.append(Check("regression", "note", "the baseline report did not change, nothing to compare"))
    else:
        checks += check_regression(baseline.report, previous.report, baseline.accepted)
    return checks


def passed(checks: list[Check]) -> bool:
    return not any(check.status == "fail" for check in checks)


def render(baseline: Baseline, checks: list[Check]) -> str:
    lines = [f"eval gate on {baseline.report_path}"]
    lines += [f"  {check.status.upper():<9}{check.kind:<12}{check.text}" for check in checks]
    failed = sum(check.status == "fail" for check in checks)
    lines.append("gate passed" if not failed else f"gate failed: {failed} check(s)")
    return "\n".join(lines)
