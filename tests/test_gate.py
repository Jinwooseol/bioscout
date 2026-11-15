"""The eval gate, on small made-up reports: what counts as fresh, as a floor, and as a regression."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from bioscout.evals import gate
from bioscout.evals.datasets import EvalItem, datasets_sha
from bioscout.evals.gate import Baseline, Code

ROOT = Path(__file__).parent.parent
ITEMS = [EvalItem(id=f"q{n}", dataset="pubmedqa", question=f"Question {n}?") for n in range(60)]
CODE = Code(prompts={"B": "bbb", "C": "ccc", "D": "ddd"}, model="m", effort="medium", judge_sha="jjj",
            datasets={"pubmedqa": ITEMS})  # fmt: skip


def report(run_id: str = "run-1", **changes: Any) -> dict[str, Any]:
    row = {"variant": "C", "dataset": "pubmedqa", "n": 60, "errors": 0, "fabricated_pmid_rate": 0.0,
           "accuracy": 0.80, "gold_in_context_rate": 0.90, "citation_rate": 1.0,
           "citation_precision": 0.86, "citation_coverage": 0.93}  # fmt: skip
    meta = {"run_id": run_id, "limit": 0, "model": "m", "effort": "medium", "datasets": ["pubmedqa"],
            "items": 60, "variants": {"C": {"name": "Agent", "prompt_sha": "ccc"}},
            "judge": {"prompt_sha": "jjj", "sentences": 300}}  # fmt: skip
    for key, value in changes.items():
        (row if key in row else meta)[key] = value
    return {"meta": meta, "rows": [row]}


def statuses(checks: list[gate.Check], kind: str) -> list[str]:
    return [check.status for check in checks if check.kind == kind]


def test_a_report_made_with_the_current_code_is_fresh_and_unmeasured_variants_are_only_noted():
    checks = gate.check_fresh(report(), CODE)
    assert gate.passed(checks)
    notes = [check.text for check in checks if check.status == "note"]
    assert notes == [
        "variant B has not been measured, so it is not gated",
        "variant D has not been measured, so it is not gated",
    ]


def test_a_change_to_the_agent_the_model_the_judge_or_the_questions_makes_the_report_stale():
    def failures(code: Code = CODE, **changes: Any) -> list[str]:
        return [c.text for c in gate.check_fresh(report(**changes), code) if c.status == "fail"]

    changed_prompt = Code(**{**CODE.__dict__, "prompts": {"C": "new"}})
    assert "variant C changed since it was measured (ccc in the report, new in the code)" in "".join(
        failures(changed_prompt)
    )
    assert "but the code now uses m (medium)" in failures(model="other")[0]
    assert "the judge prompt changed" in failures(judge={"prompt_sha": "old", "sentences": 300})[0]
    assert "covers only the first 10 questions" in failures(limit=10)[0]
    assert "the datasets have 60 questions but 61 were measured" in failures(items=61)[0]
    # A report that recorded a hash of its questions is compared by content, not by count.
    assert not failures(datasets_sha=datasets_sha(ITEMS))
    relabelled = [ITEMS[0].model_copy(update={"answer": "no"}), *ITEMS[1:]]
    stale = Code(**{**CODE.__dict__, "datasets": {"pubmedqa": relabelled}})
    assert failures(stale, datasets_sha=datasets_sha(ITEMS)) == [
        "the datasets changed since they were measured: run the eval again"
    ]


def test_a_fabricated_pmid_or_a_failed_question_fails_the_floor():
    assert statuses(gate.check_floors(report()), "floor") == ["pass"]
    (fabricated,) = gate.check_floors(report(fabricated_pmid_rate=0.01))
    assert fabricated.status == "fail" and "fabricated PMIDs in pubmedqa" in fabricated.text
    (failed,) = gate.check_floors(report(errors=2))
    assert failed.status == "fail" and "2 question(s) failed" in failed.text


def test_a_drop_beyond_the_limit_fails_unless_it_was_accepted_with_a_reason():
    before = report("run-1")
    within = gate.check_regression(report("run-2", citation_precision=0.84, accuracy=0.76), before, {})
    assert statuses(within, "regression") == ["pass"]  # 2 and 4 points, inside the 3 and 5 point limits

    beyond = gate.check_regression(report("run-2", citation_precision=0.80), before, {})
    (failure,) = [check for check in beyond if check.status == "fail"]
    assert failure.text == ("C/pubmedqa/citation_precision: 86.0% -> 80.0%, a drop of 6.0 points (limit 3)")

    waiver = {"C/pubmedqa/citation_precision": "traded for coverage"}
    accepted = gate.check_regression(report("run-2", citation_precision=0.80), before, waiver)
    assert gate.passed(accepted) and "accepted: traded for coverage" in accepted[-1].text


def test_small_groups_and_missing_metrics_are_not_gated():
    small = gate.check_regression(report("run-2", n=15, accuracy=0.40), report("run-1", n=15), {})
    assert [check.status for check in small] == ["note"] and "too small a group" in small[0].text
    no_judge = gate.check_regression(report("run-2", citation_precision=None), report("run-1"), {})
    assert gate.passed(no_judge)


def test_the_gate_compares_with_the_previous_baseline_only_when_the_report_changed():
    now = Baseline("evals/reports/run-2.json", report("run-2", accuracy=0.60))
    same = Baseline("evals/reports/run-2.json", report("run-2"))
    earlier = Baseline("evals/reports/run-1.json", report("run-1"))

    assert gate.passed(gate.run_gate(now, CODE))  # nothing to compare with
    assert gate.passed(gate.run_gate(now, CODE, same))
    checks = gate.run_gate(now, CODE, earlier)
    assert not gate.passed(checks)
    assert gate.render(now, checks).splitlines()[-1] == "gate failed: 1 check(s)"


def test_the_committed_baseline_names_a_full_report_that_exists():
    text = (ROOT / "evals" / "baseline.json").read_text(encoding="utf-8")
    baseline = Baseline.load(text, ROOT)
    assert baseline.report["meta"]["limit"] == 0 and baseline.report["rows"]
    assert set(json.loads(text)) <= {"report", "accepted"}
