"""Cost and latency: steps recorded in answers, and the tables a report builds from them."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bioscout.agent import ResearchAgent
from bioscout.answer import CitedAnswer, CitedSentence, Step, Usage
from bioscout.evals.costs import render_costs, summarize_steps
from bioscout.evals.datasets import EvalItem
from bioscout.evals.metrics import ItemScores, Record, summarize_group
from bioscout.evals.runner import write_report
from bioscout.evals.variants import VARIANTS, Context
from tests.fakes import FakeLLM, FakePubMed, reply


def usage(
    usd: float = 0.0, ms: int = 0, llm: int = 0, cached: int = 0, tools: int = 0, tokens: int = 0
) -> Usage:
    return Usage(
        usd=usd, latency_ms=ms, llm_calls=llm, cached_llm_calls=cached, tool_calls=tools, input_tokens=tokens
    )


def record(variant: str, item_id: str, total: Usage, steps: list[Step] | None = None) -> Record:
    answer = CitedAnswer(
        question="Q?", sentences=[CitedSentence(text="X.", citations=[])], usage=total, steps=steps or []
    )
    scores = ItemScores(sentences=1, cited_sentences=0, cited=0, fabricated=0)
    return Record(item_id=item_id, dataset="pubmedqa", variant=variant, answer=answer, scores=scores)


def test_new_spend_counts_the_steps_that_ran_when_only_part_of_an_answer_was_replayed():
    # Variant E after C: the plan call is replayed from C's run, the answer call is new.
    steps = [
        Step(name="plan", usage=usage(usd=0.02, ms=1, llm=1, cached=1)),
        Step(name="rank", usage=usage(ms=1800)),
        Step(name="answer", usage=usage(usd=0.04, ms=6000, llm=1)),
    ]
    partly = record("E", "partly", usage(usd=0.06, ms=7801, llm=2, cached=1), steps)
    # A verify step with four judge calls, three of them replayed: a quarter of it is new.
    mixed = [Step(name="verify", usage=usage(usd=0.08, ms=900, llm=4, cached=3))]
    verified = record("E", "verified", usage(usd=0.08, ms=900, llm=4, cached=3), mixed)
    row = summarize_group([partly, verified])

    assert row["usd_total"] == pytest.approx(0.14) and row["usd_live"] == pytest.approx(0.04 + 0.02)
    assert row["latency_n"] == 0  # neither answer ran from start to end, so neither is timed as a whole


def test_latency_and_new_spend_leave_out_replayed_answers():
    live = [record("C", f"live-{i}", usage(usd=0.10, ms=1000 * (i + 1), llm=3, tokens=900)) for i in range(3)]
    replayed = [record("C", "replayed", usage(usd=0.10, ms=2, llm=3, cached=3, tokens=900))]
    row = summarize_group(live + replayed)

    assert row["usd_per_query"] == pytest.approx(0.10) and row["usd_total"] == pytest.approx(0.40)
    assert row["usd_live"] == pytest.approx(0.30)  # the replayed answer cost nothing this time
    assert (row["latency_p50_ms"], row["latency_n"]) == (2000.0, 3)  # the 2 ms replay is not in the median
    assert row["latency_p95_ms"] is None  # too few timed answers for a p95
    assert row["input_tokens_per_query"] == 900 and row["llm_calls_per_query"] == 3


def test_p95_is_reported_from_twenty_timed_answers():
    records = [record("A", f"i{i}", usage(usd=0.01, ms=100 * (i + 1), llm=1)) for i in range(20)]
    assert summarize_group(records)["latency_p95_ms"] == 1900.0


async def test_agent_answers_record_each_step_and_the_steps_add_up_to_the_total():
    plan = {"searches": [{"purpose": "all", "query": "q1"}]}
    screen = {"relevant_pmids": ["10000001"], "enough": True, "missing": "", "next_queries": []}
    llm = FakeLLM([plan, screen, reply("yes", ("X beat Y.", ["10000001"]))])
    answer = await ResearchAgent(llm, FakePubMed()).answer("Is X better than Y?")

    assert [s.name for s in answer.steps] == ["plan", "search", "screen", "answer"]
    assert [(s.usage.llm_calls, s.usage.tool_calls) for s in answer.steps] == [(1, 0), (0, 3), (1, 0), (1, 0)]
    assert sum(s.usage.usd for s in answer.steps) == pytest.approx(answer.usage.usd)
    assert sum(s.usage.tool_calls for s in answer.steps) == answer.usage.tool_calls
    assert all(s.usage.latency_ms >= 0 for s in answer.steps)


async def test_variants_a_and_b_record_their_steps():
    item = EvalItem(id="pqa-1", dataset="pubmedqa", question="Q?", answer="yes", pmid="10000001")
    a = await VARIANTS["A"].run(item, Context(FakeLLM([reply("yes", ("X.", []))]), FakePubMed()))
    b = await VARIANTS["B"].run(item, Context(FakeLLM([reply("yes", ("X.", ["10000001"]))]), FakePubMed()))
    assert [(s.name, s.usage.llm_calls, s.usage.tool_calls) for s in a.steps] == [("answer", 1, 0)]
    assert [(s.name, s.usage.llm_calls, s.usage.tool_calls) for s in b.steps] == [
        ("search", 0, 2),
        ("answer", 1, 0),
    ]


def test_steps_are_summed_per_answer_and_timed_only_where_they_ran():
    first = record(
        "D",
        "one",
        usage(usd=0.30, llm=4),
        steps=[
            Step(name="search", usage=usage(ms=400, tools=3)),
            Step(name="verify", usage=usage(usd=0.10, ms=1000, llm=2)),
            Step(name="verify", usage=usage(usd=0.10, ms=500, llm=1)),  # the re-check after a revision
            Step(name="revise", usage=usage(usd=0.10, ms=800, llm=1)),
        ],
    )
    second = record(
        "D",
        "two",
        usage(usd=0.10, llm=2, cached=2),
        steps=[
            Step(name="search", usage=usage(ms=1, tools=3)),  # its answer was replayed, so this is not timed
            Step(name="verify", usage=usage(usd=0.10, ms=3, llm=2, cached=2)),
        ],
    )
    rows = {
        row["step"]: row
        for row in summarize_steps([first, second, record("A", "old", usage(usd=0.01, llm=1))])
    }

    assert list(rows) == ["search", "verify", "revise"] and all(r["variant"] == "D" for r in rows.values())
    verify = rows["verify"]
    assert verify["runs_per_query"] == 1.5 and verify["llm_calls_per_query"] == 2.5
    assert verify["usd_per_query"] == pytest.approx(0.15) and verify["usd_share"] == pytest.approx(0.75)
    assert (verify["latency_p50_ms"], verify["latency_n"]) == (1500.0, 1)  # both runs of the live answer
    assert (rows["search"]["latency_p50_ms"], rows["search"]["latency_n"]) == (400.0, 1)
    assert rows["revise"]["runs_per_query"] == 0.5 and rows["revise"]["usd_share"] == pytest.approx(0.25)


def test_report_has_the_cost_tables_and_keeps_latency_out_of_the_metrics_table(tmp_path: Path):
    steps = [Step(name="answer", usage=usage(usd=0.02, ms=900, llm=1))]
    records = [record("A", "pqa-1", usage(usd=0.02, ms=900, llm=1, tokens=500), steps=steps)]
    run_dir = tmp_path / "runs" / "t1"
    run_dir.mkdir(parents=True)
    (run_dir / "records.jsonl").write_text(
        "".join(r.model_dump_json() + "\n" for r in records), encoding="utf-8"
    )
    (run_dir / "meta.json").write_text(
        json.dumps({"run_id": "t1", "model": "m", "effort": "e"}), encoding="utf-8"
    )

    report = write_report(run_dir, tmp_path / "reports")
    text = report.with_suffix(".md").read_text(encoding="utf-8")
    metrics, costs = text.split("## Cost and latency\n")
    assert "p50" not in metrics and "$/query" in metrics
    assert "| A | pubmedqa | 1 | $0.020 | $0.02 | $0.02 | 500 | 0 | 1.00 | 0.00 | 0.9s | — | 1 |" in costs
    assert (
        "## Cost and latency by step" in costs
        and "| A | answer | 1.00 | 1.00 | 0.00 | $0.020 | 100.0% |" in costs
    )
    saved = json.loads(report.read_text(encoding="utf-8"))
    assert saved["steps"][0]["step"] == "answer" and saved["rows"][0]["latency_n"] == 1


def test_a_run_without_recorded_steps_has_no_step_table():
    row = {
        "variant": "A",
        "dataset": "pubmedqa",
        **summarize_group([record("A", "old", usage(usd=0.01, llm=1))]),
    }
    assert "by step" not in render_costs([row], summarize_steps([record("A", "old", usage(usd=0.01, llm=1))]))
