"""Failure classification from records: where an expected paper was lost, and the kind of a wrong answer."""

from __future__ import annotations

from bioscout.answer import CitedAnswer, CitedSentence, SearchLog
from bioscout.evals.datasets import EvalItem
from bioscout.evals.failures import paper_stage, render_failures, summarize_failures, wrong_kind
from bioscout.evals.metrics import Record


def answer(decision: str | None, cited: list[str], context: list[str], found: list[str]) -> CitedAnswer:
    return CitedAnswer(
        question="Q?",
        decision=decision,
        sentences=[CitedSentence(text="A statement.", citations=cited)],
        context_pmids=context,
        searches=[SearchLog(query="q", pmids=found, total_found=len(found))],
    )


def test_a_paper_is_counted_at_the_first_step_that_lost_it():
    full = answer("yes", cited=["10000001"], context=["10000001", "10000002"], found=["10000001", "10000002"])
    assert paper_stage(full, "10000001") == "cited"
    assert paper_stage(full, "10000002") == "not_cited"
    assert (
        paper_stage(answer("yes", [], ["10000002"], ["10000001", "10000002"]), "10000001") == "not_selected"
    )
    assert paper_stage(full, "10000009") == "not_found"


def test_a_wrong_answer_is_opposite_hedged_committed_or_missing():
    assert wrong_kind("yes", "yes") is None
    assert wrong_kind("yes", "no") == "opposite" and wrong_kind("no", "yes") == "opposite"
    assert wrong_kind("yes", "maybe") == "hedged"
    assert wrong_kind("maybe", "no") == "committed"
    assert wrong_kind("no", "not_applicable") == "no_answer" and wrong_kind("no", None) == "no_answer"


def test_the_tables_count_lost_papers_and_kinds_of_wrong_answers_per_variant():
    items = {
        "q1": EvalItem(id="q1", dataset="pubmedqa", question="Q1?", answer="yes", pmid="10000001"),
        "q2": EvalItem(id="q2", dataset="pubmedqa", question="Q2?", answer="no", pmid="10000002"),
        "d1": EvalItem(id="d1", dataset="domain", question="D1?", must_cite=["10000003", "10000004"]),
    }
    records = [
        # The agent read the source paper and still hedged, then missed the other question's paper.
        Record(item_id="q1", dataset="pubmedqa", variant="C",
               answer=answer("maybe", ["10000001"], ["10000001"], ["10000001"])),
        Record(item_id="q2", dataset="pubmedqa", variant="C",
               answer=answer("yes", [], ["10000007"], ["10000007"])),
        Record(item_id="d1", dataset="domain", variant="C",
               answer=answer("not_applicable", ["10000003"], ["10000003"], ["10000003", "10000004"])),
        # The model alone cites from memory and never searches.
        Record(item_id="q1", dataset="pubmedqa", variant="A",
               answer=CitedAnswer(question="Q1?", decision="yes")),
        Record(item_id="q2", dataset="pubmedqa", variant="A", error="boom"),
    ]  # fmt: skip
    failures = summarize_failures(records, items)

    papers = {(row["variant"], row["dataset"]): row for row in failures["papers"]}
    assert papers[("C", "pubmedqa")] == {
        "variant": "C", "dataset": "pubmedqa", "papers": 2,
        "cited": 1, "not_cited": 0, "not_selected": 0, "not_found": 1,
    }  # fmt: skip
    assert papers[("C", "domain")]["cited"] == 1 and papers[("C", "domain")]["not_selected"] == 1
    assert papers[("A", "pubmedqa")] == {
        "variant": "A", "dataset": "pubmedqa", "papers": 1,
        "cited": 0, "not_cited": None, "not_selected": None, "not_found": None,
    }  # fmt: skip

    (agent,) = [row for row in failures["answers"] if row["variant"] == "C"]
    assert (agent["n"], agent["wrong"], agent["hedged"], agent["opposite"]) == (2, 2, 1, 1)
    assert agent["wrong_with_source"] == 1  # only the hedge was written with the source paper read
    (alone,) = [row for row in failures["answers"] if row["variant"] == "A"]
    assert (alone["n"], alone["wrong"], alone["wrong_with_source"]) == (1, 0, None)

    text = render_failures(failures)
    assert "## Where answers go wrong" in text
    assert "| C | pubmedqa | 2 | 2 | 1 | 1 | 0 | 0 | 1 |" in text
    assert render_failures({"papers": [], "answers": []}) == ""
