"""Variants C and D as a LangGraph state graph.

C: plan -> search -> screen (-> search once more) -> answer.
D: the same, then verify -> (revise -> verify) -> finish, which checks every sentence against the
abstracts it cites, rewrites what is not fully backed once, and removes what nothing backs.

Every LLM step is a single JSON call through bioscout.llm, so a run replays from the response cache
and each step can be tested with a scripted LLM.
"""

from __future__ import annotations

import asyncio
import operator
import time
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import asdict, dataclass
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from bioscout import tracing
from bioscout.agent.prompts import PLAN_SCHEMA, SCREEN_SCHEMA, SYSTEM_PLAN, SYSTEM_SCREEN
from bioscout.agent.rank import Ranker
from bioscout.answer import CitedAnswer, CitedSentence, DroppedSentence, SearchLog, Step, Usage, Verdict
from bioscout.llm import LLM
from bioscout.models import AbstractBatch, ArticleAbstract
from bioscout.prompts import ANSWER_SCHEMA, SYSTEM_GROUNDED, answer_from_result, format_abstracts
from bioscout.sources.pubmed import Literature
from bioscout.verifier import (
    JudgeRefusedError,
    SentenceJudgment,
    VerifyConfig,
    apply_judgment,
    judge_sentence,
    revise_sentences,
    unbacked_note,
)


@dataclass(frozen=True)
class AgentConfig:
    max_queries: int = 4  # from the plan
    search_question: bool = True  # also run variant B's search: the question as typed
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
    steps: Annotated[list[Step], operator.add]  # one entry per node that ran, with its cost and time
    refused: bool
    answer: CitedAnswer
    # Variant D: the answer's sentences while the verify step works on them.
    work: list[CitedSentence | None]  # None marks a sentence that was removed
    pending: list[int]  # indexes into `work` that still need a verdict
    notes: dict[int, str]  # what the judge found unbacked, by index
    revisions: int
    dropped: Annotated[list[DroppedSentence], operator.add]


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(v for v in values if v))


Node = Callable[["AgentState"], Awaitable["AgentState"]]


def _timed(name: str, node: Node) -> Node:
    """Wrap a graph node so the answer records what the step cost and how long it took."""

    async def run(state: AgentState) -> AgentState:
        start = time.perf_counter()
        with tracing.span(name, kind=tracing.STEP_KINDS.get(name, "chain")) as current:
            update = await node(state)
            step = Step.since(name, start, update.get("usages", []))
            usage = step.usage
            current.set(usd=usage.usd, llm_calls=usage.llm_calls, tool_calls=usage.tool_calls)
        return {**update, "steps": [step]}

    return run


def _traced_output(answer: CitedAnswer) -> dict[str, Any]:
    """The answer as the trace shows it: what was said, on which papers, and what was removed."""
    return {
        "decision": answer.decision,
        "refused": answer.refused,
        "sentences": [
            {"text": s.text, "citations": s.citations, "verdict": s.verdict} for s in answer.sentences
        ],
        "dropped": [{"text": d.text, "reason": d.reason} for d in answer.dropped],
    }


def _interleave(ranked: list[list[str]], exclude: set[str]) -> list[str]:
    """First hit of every query, then the second of every query, and so on, skipping seen PMIDs."""
    depth = max((len(r) for r in ranked), default=0)
    return _unique(r[i] for i in range(depth) for r in ranked if i < len(r) and r[i] not in exclude)


class ResearchAgent:
    def __init__(
        self,
        llm: LLM,
        pubmed: Literature,
        config: AgentConfig | None = None,
        verify: VerifyConfig | None = None,
        ranker: Ranker | None = None,
    ) -> None:
        self.llm = llm
        self.pubmed = pubmed
        self.config = config or AgentConfig()
        self.verify = verify  # None: variant C, the answer is returned as written
        self.ranker = ranker  # None: the model screens the candidates; else variant E ranks them
        self.graph = self._build()

    @staticmethod
    def fingerprint(config: AgentConfig | None = None) -> tuple[Any, ...]:
        """Everything besides the answer prompt that changes behaviour; part of the variant's prompt_sha."""
        return (SYSTEM_PLAN, PLAN_SCHEMA, SYSTEM_SCREEN, SCREEN_SCHEMA, asdict(config or AgentConfig()))

    @staticmethod
    def rank_fingerprint(ranker: str, config: AgentConfig | None = None) -> tuple[Any, ...]:
        """The same for an agent that ranks the candidates instead of screening them (variant E)."""
        return (SYSTEM_PLAN, PLAN_SCHEMA, asdict(config or AgentConfig()), "rank", ranker)

    def _build(self, answering: bool = True) -> Any:
        """The whole graph, or with `answering` off the graph up to the choice of abstracts."""
        graph = StateGraph(AgentState)
        graph.add_node("plan", _timed("plan", self._plan))
        graph.add_node("search", _timed("search", self._search))
        # The abstracts for the answer are chosen by the model (screen) or by a ranker (rank).
        choose = "screen" if self.ranker is None else "rank"
        graph.add_node(choose, _timed(choose, self._screen if self.ranker is None else self._rank))
        graph.add_edge(START, "plan")
        graph.add_conditional_edges("plan", self._after_plan, ["search", END])
        graph.add_edge("search", choose)
        if not answering:
            stops = {"search": "search", "answer": END, END: END}
            graph.add_conditional_edges(choose, self._after_screen, stops)
            return graph.compile(name="bioscout-select")
        graph.add_node("answer", _timed("answer", self._answer))
        graph.add_conditional_edges(choose, self._after_screen, ["search", "answer", END])
        if self.verify is None:
            graph.add_edge("answer", END)
        else:
            graph.add_node("verify", _timed("verify", self._verify))
            graph.add_node("revise", _timed("revise", self._revise))
            graph.add_node("finish", self._finish)
            graph.add_edge("answer", "verify")
            graph.add_conditional_edges("verify", self._after_verify, ["revise", "finish"])
            graph.add_edge("revise", "verify")
            graph.add_edge("finish", END)
        return graph.compile(name="bioscout-agent")

    async def answer(self, question: str, on_step: Callable[[Step], None] | None = None) -> CitedAnswer:
        """Answer the question. `on_step` is called as each step finishes, with what it cost and how
        long it took; the demo shows progress with it."""
        answer, _ = await self.run(question, on_step)
        return answer

    @staticmethod
    def _start(question: str) -> AgentState:
        return {
            "question": question,
            "round": 0,
            "searches": [],
            "abstracts": {},
            "selected": [],
            "usages": [],
            "steps": [],
            "notes": {},
            "revisions": 0,
            "dropped": [],
        }

    async def select(self, question: str) -> AgentState:
        """Run up to the choice of abstracts and stop, writing no answer. The state that comes back holds
        the searches, every abstract fetched and the PMIDs chosen (see evals/selection.py)."""
        with tracing.span("bioscout.select", kind="agent") as trace:
            trace.input(question)
            state: AgentState = await self._build(answering=False).ainvoke(self._start(question))
            trace.output({"selected": state.get("selected", [])})
        return state

    async def run(
        self, question: str, on_step: Callable[[Step], None] | None = None
    ) -> tuple[CitedAnswer, dict[str, ArticleAbstract]]:
        """`answer`, plus every abstract the agent fetched on the way, by PMID."""
        start = self._start(question)
        state = start
        reported = 0
        with tracing.span("bioscout.answer", kind="agent", verify=self.verify is not None) as trace:
            trace.input(question)
            async for state in self.graph.astream(start, stream_mode="values"):  # the state after each step
                steps = state.get("steps", [])
                for step in steps[reported:]:
                    if on_step is not None:
                        on_step(step)
                reported = len(steps)
            if "answer" in state:
                answer = state["answer"].model_copy(update={"steps": state["steps"]})
            else:  # a step before the answer was refused, even after the server-side fallback
                answer = CitedAnswer(
                    question=question,
                    searches=state["searches"],
                    usage=Usage.total(state["usages"]),
                    steps=state["steps"],
                    refused=True,
                )
            trace.output(_traced_output(answer))
            trace.set(
                usd=answer.usage.usd,
                sentences=len(answer.sentences),
                dropped=len(answer.dropped),
                searches=len(answer.searches),
            )
        return answer, state["abstracts"]

    async def _plan(self, state: AgentState) -> AgentState:
        result = await self.llm.json(
            system=SYSTEM_PLAN, user=f"Question: {state['question']}", schema=PLAN_SCHEMA
        )
        if result.refused or result.data is None:
            return {"usages": [result.usage], "refused": True}
        planned = _unique(s["query"].strip() for s in result.data["searches"])[: self.config.max_queries]
        # Broad MeSH queries can bury a paper that the question's own wording finds at once, so the first
        # round also runs variant B's search. An empty plan degrades to that search rather than to none.
        first = [state["question"]] if self.config.search_question or not planned else []
        return {"usages": [result.usage], "pending_queries": _unique([*first, *planned])}

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
        question = state["question"]
        # The question's own hits are variant B's whole context: they go first so the cap never drops them,
        # and C screens every abstract B would answer from.
        lead = [p for log in logs if log.query == question for p in log.pmids if p not in known]
        rest = _interleave([log.pmids for log in logs if log.query != question], exclude=set(known))
        new = _unique([*lead, *rest])[: self.config.max_new_candidates]
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

    async def _rank(self, state: AgentState) -> AgentState:
        """Variant E: keep the candidates closest to the question. No model call, so no second round."""
        assert self.ranker is not None
        candidates = [state["abstracts"][pmid] for pmid in state["fresh"]]
        ranked = await self.ranker.rank(state["question"], candidates)
        selected = [pmid for pmid, _ in ranked][: self.config.max_context]
        return {"selected": selected, "enough": True, "pending_queries": [], "usages": []}

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
        return {
            "answer": answer,
            "usages": [result.usage],
            "work": list(answer.sentences),
            "pending": list(range(len(answer.sentences))),
        }

    async def _verify(self, state: AgentState) -> AgentState:
        """Judge the sentences that have no verdict yet against the abstracts they cite."""
        work = list(state["work"])

        async def judge(index: int) -> tuple[SentenceJudgment | None, Usage]:
            sentence = work[index]
            assert sentence is not None
            try:
                return await judge_sentence(
                    self.llm, state["question"], sentence.text, sentence.citations, state["abstracts"]
                )
            except JudgeRefusedError as exc:
                return None, exc.usage  # the sentence stays as written, without a verdict

        results = await asyncio.gather(*(judge(i) for i in state["pending"]))
        notes = dict(state["notes"])
        for index, (judgment, _) in zip(state["pending"], results, strict=True):
            sentence = work[index]
            if judgment is not None and sentence is not None:
                notes[index] = unbacked_note(sentence, judgment)
                work[index] = apply_judgment(sentence, judgment)
        return {"work": work, "pending": [], "notes": notes, "usages": [usage for _, usage in results]}

    @staticmethod
    def _unbacked(state: AgentState) -> dict[int, CitedSentence]:
        """Sentences the judge did not find fully backed. One it could not check is not among them."""
        return {
            i: s
            for i, s in enumerate(state["work"])
            if s is not None and s.verdict in (Verdict.PARTIAL, Verdict.UNSUPPORTED)
        }

    def _after_verify(self, state: AgentState) -> str:
        assert self.verify is not None
        if self._unbacked(state) and state["revisions"] < self.verify.max_revisions:
            return "revise"
        return "finish"

    async def _revise(self, state: AgentState) -> AgentState:
        """Rewrite what is not fully backed from the abstracts the answer was written from."""
        context = [state["abstracts"][p] for p in state["selected"]]
        revised, usage = await revise_sentences(
            self.llm, state["question"], self._unbacked(state), state["notes"], context
        )
        work = list(state["work"])
        dropped: list[DroppedSentence] = []
        pending: list[int] = []
        for index, sentence in revised.items():
            old = work[index]
            assert old is not None
            if sentence is None:  # the reviser found nothing in the abstracts to keep
                dropped.append(DroppedSentence(text=old.text, citations=old.citations, reason="not_backed"))
            else:
                pending.append(index)
            work[index] = sentence
        return {
            "work": work,
            "pending": pending,
            "dropped": dropped,
            "revisions": state["revisions"] + 1,
            "usages": [usage],
        }

    async def _finish(self, state: AgentState) -> AgentState:
        """The verified answer: sentences that nothing backs are removed, the rest carry their verdict."""
        kept: list[CitedSentence] = []
        dropped = list(state["dropped"])
        for sentence in state["work"]:
            if sentence is None:
                continue
            if sentence.verdict is Verdict.UNSUPPORTED:
                reason = "unsupported"
                dropped.append(
                    DroppedSentence(text=sentence.text, citations=sentence.citations, reason=reason)
                )
            else:
                kept.append(sentence)
        update = {"sentences": kept, "dropped": dropped, "usage": Usage.total(state["usages"])}
        return {"answer": state["answer"].model_copy(update=update)}
