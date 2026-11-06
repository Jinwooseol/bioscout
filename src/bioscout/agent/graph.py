"""Variant C: plan -> search -> screen (-> search once more) -> answer, as a LangGraph state graph.

Every LLM step is a single JSON call through bioscout.llm, so a run replays from the response cache
and each step can be tested with a scripted LLM. Day 4 adds a verify node after answer (variant D).
"""

from __future__ import annotations

import asyncio
import operator
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from bioscout.agent.prompts import PLAN_SCHEMA, SCREEN_SCHEMA, SYSTEM_PLAN, SYSTEM_SCREEN
from bioscout.answer import CitedAnswer, SearchLog, Usage
from bioscout.llm import LLM
from bioscout.models import AbstractBatch, ArticleAbstract
from bioscout.prompts import ANSWER_SCHEMA, SYSTEM_GROUNDED, answer_from_result, format_abstracts
from bioscout.sources.pubmed import Literature


@dataclass(frozen=True)
class AgentConfig:
    max_queries: int = 4  # from the plan
    results_per_query: int = 10
    max_new_candidates: int = 25  # abstracts shown to the screener per round
    max_rounds: int = 2  # search rounds; the second runs only when the screener asks for it
    max_next_queries: int = 3
    max_context: int = 10  # abstracts given to the answer step, the same budget as variant B


class AgentState(TypedDict, total=False):
    question: str
    round: int
    pending_queries: list[str]
    searches: Annotated[list[SearchLog], operator.add]
    abstracts: dict[str, ArticleAbstract]  # every abstract fetched, by PMID
    fresh: list[str]  # PMIDs fetched in the latest round, not yet screened
    selected: list[str]  # the screener's picks, most useful first
    enough: bool
    usages: Annotated[list[Usage], operator.add]
    refused: bool
    answer: CitedAnswer


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(v for v in values if v))


def _interleave(ranked: list[list[str]], exclude: set[str]) -> list[str]:
    """First hit of every query, then the second of every query, and so on, skipping seen PMIDs."""
    depth = max((len(r) for r in ranked), default=0)
    return _unique(r[i] for i in range(depth) for r in ranked if i < len(r) and r[i] not in exclude)


class ResearchAgent:
    def __init__(self, llm: LLM, pubmed: Literature, config: AgentConfig | None = None) -> None:
        self.llm = llm
        self.pubmed = pubmed
        self.config = config or AgentConfig()
        self.graph = self._build()

    @staticmethod
    def fingerprint(config: AgentConfig | None = None) -> tuple[Any, ...]:
        """Everything besides the answer prompt that changes behaviour; part of the variant's prompt_sha."""
        return (SYSTEM_PLAN, PLAN_SCHEMA, SYSTEM_SCREEN, SCREEN_SCHEMA, asdict(config or AgentConfig()))

    def _build(self) -> Any:
        graph = StateGraph(AgentState)
        graph.add_node("plan", self._plan)
        graph.add_node("search", self._search)
        graph.add_node("screen", self._screen)
        graph.add_node("answer", self._answer)
        graph.add_edge(START, "plan")
        graph.add_conditional_edges("plan", self._after_plan, ["search", END])
        graph.add_edge("search", "screen")
        graph.add_conditional_edges("screen", self._after_screen, ["search", "answer", END])
        graph.add_edge("answer", END)
        return graph.compile(name="bioscout-agent")

    async def answer(self, question: str) -> CitedAnswer:
        state: AgentState = await self.graph.ainvoke(
            {"question": question, "round": 0, "searches": [], "abstracts": {}, "selected": [], "usages": []}
        )
        if "answer" in state:
            return state["answer"]
        # A step before the answer was refused, even after the server-side fallback.
        return CitedAnswer(
            question=question, searches=state["searches"], usage=Usage.total(state["usages"]), refused=True
        )

    async def _plan(self, state: AgentState) -> AgentState:
        result = await self.llm.json(
            system=SYSTEM_PLAN, user=f"Question: {state['question']}", schema=PLAN_SCHEMA
        )
        if result.refused or result.data is None:
            return {"usages": [result.usage], "refused": True}
        queries = _unique(s["query"].strip() for s in result.data["searches"])[: self.config.max_queries]
        # An empty plan degrades to variant B's search rather than to no search at all.
        return {"usages": [result.usage], "pending_queries": queries or [state["question"]]}

    def _after_plan(self, state: AgentState) -> str:
        return END if state.get("refused") else "search"

    async def _search(self, state: AgentState) -> AgentState:
        queries = state["pending_queries"]
        results = await asyncio.gather(
            *(self.pubmed.search(q, max_results=self.config.results_per_query) for q in queries)
        )
        logs = [
            SearchLog(query=q, pmids=[a.pmid for a in r.articles], total_found=r.total_found)
            for q, r in zip(queries, results, strict=True)
        ]
        known = state["abstracts"]
        new = _interleave([log.pmids for log in logs], exclude=set(known))[: self.config.max_new_candidates]
        batch = await self.pubmed.abstracts(new) if new else AbstractBatch(articles=[])
        fetched = {a.pmid: a for a in batch.articles}
        return {
            "round": state["round"] + 1,
            "pending_queries": [],
            "searches": logs,
            "abstracts": known | fetched,
            "fresh": [p for p in new if p in fetched],
            "usages": [Usage(tool_calls=len(queries) + bool(new))],
        }

    async def _screen(self, state: AgentState) -> AgentState:
        cfg = self.config
        shown = _unique([*state["selected"], *state["fresh"]])  # earlier picks are re-ranked with new ones
        runs = "\n".join(f"- {s.query} ({s.total_found} hits)" for s in state["searches"])
        articles = [state["abstracts"][p] for p in shown]
        user = (
            f"Question: {state['question']}\n\n"
            f"Searches run so far (round {state['round']} of {cfg.max_rounds}):\n{runs}\n\n"
            "Candidate abstracts:\n\n"
            + (format_abstracts(articles) or "(None: the searches returned no new articles.)")
        )
        if state["round"] >= cfg.max_rounds:
            user += "\n\nThis is the last round: new queries will not be run."
        result = await self.llm.json(system=SYSTEM_SCREEN, user=user, schema=SCREEN_SCHEMA)
        if result.refused or result.data is None:
            return {"usages": [result.usage], "refused": True}

        data = result.data
        # Only abstracts the agent actually fetched can reach the answer step.
        selected = [p for p in _unique(str(p).strip() for p in data["relevant_pmids"]) if p in shown]
        already_run = {s.query for s in state["searches"]}
        next_queries = [q for q in _unique(q.strip() for q in data["next_queries"]) if q not in already_run]
        return {
            "usages": [result.usage],
            "selected": selected[: cfg.max_context],
            "enough": bool(data["enough"]),
            "pending_queries": next_queries[: cfg.max_next_queries],
        }

    def _after_screen(self, state: AgentState) -> str:
        if state.get("refused"):
            return END
        if not state["enough"] and state["pending_queries"] and state["round"] < self.config.max_rounds:
            return "search"
        return "answer"

    async def _answer(self, state: AgentState) -> AgentState:
        pmids = state["selected"]
        context = format_abstracts([state["abstracts"][p] for p in pmids])
        user = (
            f"Question: {state['question']}\n\n"
            "Abstracts selected from PubMed searches for the question, most relevant first:\n\n"
            + (context or "(The searches found no relevant abstracts.)")
        )
        result = await self.llm.json(system=SYSTEM_GROUNDED, user=user, schema=ANSWER_SCHEMA)
        usage = Usage.total([*state["usages"], result.usage])
        answer = answer_from_result(
            state["question"], result, usage=usage, searches=state["searches"], context_pmids=pmids
        )
        return {"answer": answer}
