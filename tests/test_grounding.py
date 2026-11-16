"""The citation guard: an answer written from abstracts cites only those abstracts."""

from __future__ import annotations

from bioscout.agent import ResearchAgent
from bioscout.answer import Usage
from bioscout.evals.datasets import EvalItem
from bioscout.evals.metrics import Record, render_markdown, score_item, summarize
from bioscout.evals.variants import VARIANTS
from bioscout.llm import LLMResult
from bioscout.prompts import answer_from_result, ground_citations
from tests.fakes import FakeLLM, FakePubMed, reply
from tests.test_agent import plan, screen

CONTEXT = ["10000001", "10000002"]


def test_only_the_given_pmids_are_kept_and_a_garbled_entry_is_searched_for_them():
    written = ["10000001", "24, 10000002", "99999999", "10000001", " 10000002 ", "see PMID 10000001"]
    kept, odd = ground_citations(written, CONTEXT)
    assert kept == ["10000001", "10000002"]  # in the order first cited, without repeats
    assert odd == ["24, 10000002", "99999999", "see PMID 10000001"]  # as written; the padded one was fine
    assert ground_citations(["10000001"], []) == ([], ["10000001"])  # nothing was given, nothing may be cited


def test_an_answer_written_from_abstracts_cites_no_other_paper_but_one_from_memory_is_left_alone():
    data = reply("yes", ("X works.", ["10000001", "24, 10000002"]), ("Y too.", ["99999999"]))
    result = LLMResult(data=data, usage=Usage(), stop_reason="end_turn")

    grounded = answer_from_result("Q?", result, usage=Usage(), context_pmids=CONTEXT)
    assert [s.citations for s in grounded.sentences] == [["10000001", "10000002"], []]
    assert grounded.ungrounded == ["24, 10000002", "99999999"]

    from_memory = answer_from_result("Q?", result, usage=Usage())  # variant A is given no abstracts
    assert from_memory.sentences[0].citations == ["10000001", "24, 10000002"] and not from_memory.ungrounded


async def test_the_agent_drops_a_citation_it_was_not_given_and_the_report_counts_it():
    pubmed = FakePubMed(hits=CONTEXT)
    answer_reply = reply("yes", ("X works.", ["10000001", "10000003"]))  # 10000003 exists but was not given
    llm = FakeLLM([plan("drug X"), screen(CONTEXT), answer_reply])
    answer = await ResearchAgent(llm, pubmed).answer("Is X good?")
    assert answer.cited_pmids == ["10000001"] and answer.ungrounded == ["10000003"]

    item = EvalItem(id="q1", dataset="pubmedqa", question="Is X good?", answer="yes", pmid="10000001")
    scores = await score_item(item, answer, pubmed)
    assert (scores.cited, scores.fabricated, scores.ungrounded) == (1, 0, 1)

    record = Record(item_id="q1", dataset="pubmedqa", variant="C", answer=answer, scores=scores)
    rows = summarize([record])
    assert rows[0]["ungrounded_citations"] == 1
    meta = {"run_id": "r", "model": "m", "effort": "medium"}
    assert "Citations outside the abstracts, removed" in render_markdown(rows, meta)
    clean = [row | {"ungrounded_citations": 0} for row in rows]
    assert "Citations outside the abstracts" not in render_markdown(
        clean, meta
    )  # shown only when it happened


def test_the_guard_is_part_of_the_fingerprint_of_every_variant_that_is_given_abstracts():
    assert VARIANTS["A"].prompt_sha == "42f57b304df8"  # answers from memory: unchanged by the guard
    shas = {key: variant.prompt_sha for key, variant in VARIANTS.items()}
    assert len(set(shas.values())) == len(shas)
    assert (
        "279383ac6a13" not in shas.values() and "263c2ad233de" not in shas.values()
    )  # B and C before the guard
