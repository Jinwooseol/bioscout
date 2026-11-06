"""Variant C's graph, offline: a scripted LLM answers the plan, screen and answer steps in order."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from bioscout.agent import AgentConfig, ResearchAgent
from bioscout.agent.prompts import PLAN_SCHEMA, SCREEN_SCHEMA
from bioscout.evals.datasets import EvalItem
from bioscout.evals.metrics import score_item
from bioscout.evals.variants import VARIANTS, Context
from bioscout.prompts import ANSWER_SCHEMA, SYSTEM_GROUNDED
from tests.fakes import FakeLLM, FakePubMed, reply


def plan(*queries: str) -> dict[str, Any]:
    return {"searches": [{"purpose": f"part {i}", "query": q} for i, q in enumerate(queries)]}


def screen(relevant: list[str], enough: bool = True, next_queries: list[str] | None = None) -> dict[str, Any]:
    return {
        "relevant_pmids": relevant,
        "enough": enough,
        "missing": "" if enough else "trials in humans",
        "next_queries": next_queries or [],
    }


async def test_one_round_plans_searches_screens_and_answers():
    pubmed = FakePubMed(by_query={"q1": ["10000001", "10000002"], "q2": ["10000002", "10000003"]})
    llm = FakeLLM(
        [
            plan("q1", "q2", "q1"),
            screen(["10000003", "10000001"]),
            reply("yes", ("X beat Y.", ["10000003"])),
        ]
    )
    answer = await ResearchAgent(llm, pubmed).answer("Is X better than Y?")

    assert [c["schema"] for c in llm.calls] == [PLAN_SCHEMA, SCREEN_SCHEMA, ANSWER_SCHEMA]
    assert pubmed.queries == ["q1", "q2"]  # the repeated query runs once
    assert pubmed.fetched == [["10000001", "10000002", "10000003"]]  # rank-interleaved, deduplicated
    assert "- q1 (2 hits)" in llm.calls[1]["user"]

    answer_step = llm.calls[2]
    assert answer_step["system"] == SYSTEM_GROUNDED  # the same answer prompt as variant B
    assert "PMID 10000003" in answer_step["user"] and "PMID 10000002" not in answer_step["user"]
    assert answer.context_pmids == ["10000003", "10000001"]
    assert [s.query for s in answer.searches] == ["q1", "q2"]
    assert answer.decision == "yes" and answer.cited_pmids == ["10000003"]
    assert (answer.usage.llm_calls, answer.usage.tool_calls) == (3, 3)
    assert answer.usage.usd == pytest.approx(0.03)


async def test_screener_can_ask_for_one_more_round_of_new_queries():
    pubmed = FakePubMed(by_query={"q1": ["10000001"], "broader": ["10000001", "10000002"]})
    llm = FakeLLM(
        [
            plan("q1"),
            screen([], enough=False, next_queries=["q1", "broader"]),
            screen(["10000002", "10000001"]),
            reply("maybe", ("Evidence is limited.", ["10000002"])),
        ]
    )
    answer = await ResearchAgent(llm, pubmed).answer("Q?")

    assert pubmed.queries == ["q1", "broader"]  # a query already run is not repeated
    assert pubmed.fetched == [["10000001"], ["10000002"]]  # round 2 fetches only new PMIDs
    second = llm.calls[2]["user"]
    assert "round 2 of 2" in second and "last round" in second
    # Round 1's rejected candidate is not shown again, so the screener cannot bring it back.
    assert "PMID 10000002" in second and "PMID 10000001" not in second
    assert answer.context_pmids == ["10000002"]
    assert len(answer.searches) == 2


async def test_rounds_are_capped_and_an_empty_selection_is_answered_honestly():
    llm = FakeLLM(
        [
            plan("q1"),
            screen([], enough=False, next_queries=["q2"]),
            screen([], enough=False, next_queries=["q3"]),
            reply("maybe", ("No relevant study was found.", [])),
        ]
    )
    pubmed = FakePubMed()
    answer = await ResearchAgent(llm, pubmed).answer("Q?")

    assert pubmed.queries == ["q1", "q2"]
    assert len(llm.calls) == 4
    assert "found no relevant abstracts" in llm.calls[3]["user"]
    assert answer.context_pmids == [] and answer.cited_pmids == []


async def test_zero_hits_are_screened_so_the_agent_can_broaden():
    pubmed = FakePubMed(hits=[], by_query={"broad": ["10000001"]})
    llm = FakeLLM(
        [
            plan("narrow"),
            screen([], enough=False, next_queries=["broad"]),
            screen(["10000001"]),
            reply("yes", ("X beat Y.", ["10000001"])),
        ]
    )
    answer = await ResearchAgent(llm, pubmed).answer("Q?")

    first = llm.calls[1]["user"]
    assert "- narrow (0 hits)" in first and "returned no new articles" in first
    assert pubmed.fetched == [["10000001"]]  # no abstracts call for an empty round
    assert answer.usage.tool_calls == 3  # 2 searches + 1 abstracts fetch
    assert answer.context_pmids == ["10000001"]


async def test_screener_cannot_select_pmids_it_was_not_shown():
    llm = FakeLLM([plan("q1"), screen(["99999999", " 10000002 "]), reply("yes", ("X.", ["10000002"]))])
    answer = await ResearchAgent(llm, FakePubMed()).answer("Q?")
    assert answer.context_pmids == ["10000002"]


async def test_context_is_capped_at_the_variant_b_budget():
    llm = FakeLLM([plan("q1"), screen(["10000002", "10000001"]), reply("yes", ("X.", ["10000002"]))])
    answer = await ResearchAgent(llm, FakePubMed(), AgentConfig(max_context=1)).answer("Q?")
    assert answer.context_pmids == ["10000002"]


async def test_empty_plan_falls_back_to_searching_the_question():
    pubmed = FakePubMed()
    llm = FakeLLM([{"searches": []}, screen(["10000002"]), reply("yes", ("X.", ["10000002"]))])
    await ResearchAgent(llm, pubmed).answer("Is X better than Y?")
    assert pubmed.queries == ["Is X better than Y?"]


@pytest.mark.parametrize(
    ("replies", "llm_calls", "searches"),
    [([None], 1, 0), ([plan("q1"), None], 2, 1)],
    ids=["plan refused", "screen refused"],
)
async def test_a_refusal_before_the_answer_ends_the_run(replies, llm_calls, searches):
    answer = await ResearchAgent(FakeLLM(replies), FakePubMed()).answer("Q?")
    assert answer.refused and answer.sentences == [] and answer.decision is None
    assert answer.usage.llm_calls == llm_calls and len(answer.searches) == searches


async def test_variant_c_is_scored_through_the_source_paper_funnel():
    item = EvalItem(id="pqa-10000001", dataset="pubmedqa", question="Q?", answer="yes", pmid="10000001")
    pubmed = FakePubMed()
    llm = FakeLLM([plan("q1"), screen(["10000001"]), reply("yes", ("X beat Y.", ["10000001"]))])
    answer = await VARIANTS["C"].run(item, Context(llm, pubmed))
    scores = await score_item(item, answer, pubmed)
    assert (scores.retrieved_gold, scores.gold_in_context, scores.gold_cited) == (True, True, True)
    assert scores.correct is True


def test_prompt_hashes_cover_agent_settings_and_keep_a_and_b_stable():
    # Changing A's or B's prompt makes earlier runs incomparable; update these only on purpose.
    assert VARIANTS["A"].prompt_sha == "42f57b304df8"
    assert VARIANTS["B"].prompt_sha == "279383ac6a13"
    tweaked = replace(VARIANTS["C"], fingerprint=ResearchAgent.fingerprint(AgentConfig(max_rounds=1)))
    assert len({VARIANTS["B"].prompt_sha, VARIANTS["C"].prompt_sha, tweaked.prompt_sha}) == 3
