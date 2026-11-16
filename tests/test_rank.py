"""Variant E: abstracts chosen by embedding similarity. A word-count embedder stands in for the model."""

from __future__ import annotations

import re
from collections.abc import Sequence

from bioscout.agent import AgentConfig, ResearchAgent
from bioscout.agent.prompts import PLAN_SCHEMA, SCREEN_SCHEMA
from bioscout.agent.rank import EMBED_MODEL, EmbeddingRanker, cosine, passage_text
from bioscout.evals.datasets import EvalItem
from bioscout.evals.selection import Selection, render_selection, select_one, summarize_selection
from bioscout.evals.variants import VARIANTS, Context
from bioscout.models import ArticleAbstract
from bioscout.prompts import ANSWER_SCHEMA
from tests.fakes import FakeLLM, FakePubMed, reply, synthetic_abstract
from tests.test_agent import plan, screen

TEXTS = {
    "10000001": "Drug X caused nausea in some adults.",
    "10000002": "Drug X lowered mortality: mortality fell by a third.",
    "10000003": "The cost of drug X was high.",
}
QUESTION = "Does drug X lower mortality?"


class WordEmbedder:
    """Counts a few words. Enough to make 'closest to the question' mean something in a test."""

    name = "words"
    VOCABULARY = ("mortality", "nausea", "cost", "lower", "lowered")

    def _vector(self, text: str) -> Sequence[float]:
        words = re.findall(r"[a-z]+", text.lower())
        return [float(words.count(word)) for word in self.VOCABULARY]

    def embed_query(self, text: str) -> Sequence[float]:
        return self._vector(text)

    def embed_passages(self, texts: list[str]) -> list[Sequence[float]]:
        return [self._vector(text) for text in texts]


def ranker() -> EmbeddingRanker:
    return EmbeddingRanker(WordEmbedder())


def pubmed() -> FakePubMed:
    return FakePubMed(hits=list(TEXTS), texts=TEXTS)


async def test_abstracts_are_ordered_by_similarity_to_the_question():
    articles = [synthetic_abstract(pmid, text) for pmid, text in TEXTS.items()]
    ranked = await ranker().rank(QUESTION, articles)
    assert [pmid for pmid, _ in ranked] == ["10000002", "10000001", "10000003"]  # ties keep the search order
    assert ranked[0][1] > 0.5 and ranked[1][1] == 0.0
    assert await ranker().rank(QUESTION, []) == []
    assert cosine([1.0, 0.0], [0.0, 0.0]) == 0.0  # a text with none of the words
    untitled = ArticleAbstract(pmid="10000009", title="Only a title")
    assert passage_text(untitled) == "Only a title"
    assert ranker().name == "embedding:words"


async def test_the_ranking_agent_makes_no_screening_call_and_searches_once():
    llm = FakeLLM([plan("drug X"), reply("yes", ("X lowers mortality.", ["10000002"]))])
    agent = ResearchAgent(llm, pubmed(), config=AgentConfig(max_context=2), ranker=ranker())
    answer = await agent.answer(QUESTION)

    assert [call["schema"] for call in llm.calls] == [PLAN_SCHEMA, ANSWER_SCHEMA]
    assert [step.name for step in answer.steps] == ["plan", "search", "rank", "answer"]
    assert answer.context_pmids == ["10000002", "10000001"]  # the two closest, best first
    assert len(answer.searches) == 2  # the question as typed and the planned query: one round


async def test_select_stops_at_the_choice_of_abstracts_for_either_way_of_choosing():
    ranked_llm = FakeLLM([plan("drug X")])
    state = await ResearchAgent(ranked_llm, pubmed(), ranker=ranker()).select(QUESTION)
    assert state["selected"][0] == "10000002" and "answer" not in state
    assert [call["schema"] for call in ranked_llm.calls] == [PLAN_SCHEMA]

    screened_llm = FakeLLM([plan("drug X"), screen(["10000003"])])
    state = await ResearchAgent(screened_llm, pubmed()).select(QUESTION)
    assert state["selected"] == ["10000003"] and "answer" not in state
    assert [call["schema"] for call in screened_llm.calls] == [PLAN_SCHEMA, SCREEN_SCHEMA]


async def test_variant_e_runs_with_the_rankers_name_in_its_fingerprint():
    variant = VARIANTS["E"]
    assert variant.prompt_sha != VARIANTS["C"].prompt_sha
    assert ResearchAgent.rank_fingerprint("embedding:other") != ResearchAgent.rank_fingerprint(
        f"embedding:{EMBED_MODEL}"
    )
    assert VARIANTS["C"].prompt_sha == "1c10fdf891a1"  # adding E must not change what C was measured with

    llm = FakeLLM([plan("drug X"), reply("yes", ("X lowers mortality.", ["10000002"]))])
    item = EvalItem(id="q1", dataset="pubmedqa", question=QUESTION, answer="yes", pmid="10000002")
    answer = await variant.run(item, Context(llm, pubmed(), ranker=ranker()))
    assert answer.context_pmids[0] == "10000002" and answer.decision == "yes"


async def test_the_selection_table_counts_the_expected_papers_that_were_found_and_chosen():
    item = EvalItem(id="q1", dataset="pubmedqa", question=QUESTION, answer="yes", pmid="10000002")
    agent = ResearchAgent(FakeLLM([plan("drug X")]), pubmed(), ranker=ranker())
    ranked = await select_one(item, "E", agent)
    assert ranked.selected[0] == "10000002" and ranked.usage.llm_calls == 1 and ranked.choose_ms is not None

    domain = EvalItem(
        id="d1", dataset="domain", question="D?", must_cite=["10000001", "10000003", "10000007"]
    )
    selections = [
        Selection("q1", "pubmedqa", "C", candidates=["10000001", "10000002"], selected=["10000001"]),
        Selection(
            "d1", "domain", "C", candidates=["10000001", "10000003"], selected=["10000001", "10000003"]
        ),
        ranked,
        Selection("d1", "domain", "E", candidates=["10000001", "10000003"], selected=["10000003"]),
        Selection("d2", "domain", "E", error="CacheMissError: nothing recorded"),
    ]
    items = {"q1": item, "d1": domain, "d2": domain}
    rows = {(row["variant"], row["dataset"]): row for row in summarize_selection(selections, items)}

    assert (rows[("C", "pubmedqa")]["found"], rows[("C", "pubmedqa")]["selected"]) == (1, 0)
    assert rows[("E", "pubmedqa")]["selected_rate"] == 1.0
    assert (rows[("C", "domain")]["papers"], rows[("C", "domain")]["selected"]) == (3, 2)
    e_domain = rows[("E", "domain")]
    assert (e_domain["n"], e_domain["errors"], e_domain["selected"]) == (2, 1, 1)
    assert e_domain["same_as_first"] == 0.5 and rows[("C", "domain")]["same_as_first"] is None

    meta = {
        "run_id": "r",
        "model": "m",
        "effort": "medium",
        "cache_mode": "readonly",
        "ranker": "embedding:words",
    }
    text = render_selection(list(rows.values()), meta)
    assert "| E | pubmedqa | 1 | 1 | 1 | 1 | 100.0% |" in text
