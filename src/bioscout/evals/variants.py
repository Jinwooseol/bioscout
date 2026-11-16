"""Answer variants the eval compares. All use the same model, effort and output schema, so differences
in the results come from the design (retrieval, agent, verifier), not from the model."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from bioscout.agent import ResearchAgent
from bioscout.agent.rank import EMBED_MODEL, Ranker, default_ranker
from bioscout.answer import CitedAnswer, SearchLog, Step, Usage
from bioscout.evals.datasets import EvalItem
from bioscout.llm import LLM
from bioscout.models import AbstractBatch
from bioscout.prompts import (
    ANSWER_RULES,
    ANSWER_SCHEMA,
    SYSTEM_GROUNDED,
    answer_from_result,
    format_abstracts,
)
from bioscout.sources.pubmed import Literature
from bioscout.verifier import VerifyConfig, verify_fingerprint

SYSTEM_LLM_ONLY = f"""\
You are a biomedical research assistant answering from your knowledge of the published literature.
{ANSWER_RULES}
Cite only papers you are confident exist and say what the sentence claims. If you are unsure of a \
PMID, leave "citations" empty rather than guess."""


@dataclass
class Context:
    llm: LLM
    pubmed: Literature
    ranker: Ranker | None = None  # variant E's; the default embedding ranker when not given


@dataclass(frozen=True)
class Variant:
    key: str
    name: str
    system_prompt: str  # of the step that writes the answer
    run: Callable[[EvalItem, Context], Awaitable[CitedAnswer]]
    fingerprint: tuple[Any, ...] = ()  # other prompts, schemas and settings that change behaviour

    @property
    def prompt_sha(self) -> str:
        """Changes whenever a prompt, schema or setting changes, so runs record exactly what was asked."""
        raw = json.dumps([self.system_prompt, ANSWER_SCHEMA, *self.fingerprint], sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


async def llm_only(item: EvalItem, ctx: Context) -> CitedAnswer:
    """A: no tools. Measures how often the model invents or misattributes citations on its own."""
    start = time.perf_counter()
    result = await ctx.llm.json(
        system=SYSTEM_LLM_ONLY, user=f"Question: {item.question}", schema=ANSWER_SCHEMA
    )
    answer = answer_from_result(item.question, result, usage=result.usage)
    answer.steps = [Step.since("answer", start, [result.usage])]
    return answer


async def naive_rag(item: EvalItem, ctx: Context) -> CitedAnswer:
    """B: the question verbatim as a PubMed query, top 10 abstracts as context, one LLM call."""
    start = time.perf_counter()
    found = await ctx.pubmed.search(item.question, max_results=10)
    search = SearchLog(
        query=item.question, pmids=[a.pmid for a in found.articles], total_found=found.total_found
    )
    batch = await ctx.pubmed.abstracts(search.pmids) if search.pmids else AbstractBatch(articles=[])
    tools = Usage(tool_calls=1 + bool(search.pmids))
    searched = Step.since("search", start, [tools])

    start = time.perf_counter()
    context = format_abstracts(batch.articles) or "(The PubMed search returned no articles.)"
    user = (
        f"Question: {item.question}\n\n"
        f"Abstracts from a PubMed search for the question, most relevant first:\n\n{context}"
    )
    result = await ctx.llm.json(system=SYSTEM_GROUNDED, user=user, schema=ANSWER_SCHEMA)
    answer = answer_from_result(
        item.question,
        result,
        usage=Usage.total([result.usage, tools]),
        searches=[search],
        context_pmids=[a.pmid for a in batch.articles],
    )
    answer.steps = [searched, Step.since("answer", start, [result.usage])]
    return answer


async def agent(item: EvalItem, ctx: Context) -> CitedAnswer:
    """C: plan, search, screen (and search once more if needed), then answer like B (src/bioscout/agent)."""
    return await ResearchAgent(ctx.llm, ctx.pubmed).answer(item.question)


async def verified_agent(item: EvalItem, ctx: Context) -> CitedAnswer:
    """D: C, then every sentence is checked against the abstracts it cites; what is not fully backed is
    rewritten once and what nothing backs is removed (src/bioscout/verifier)."""
    return await ResearchAgent(ctx.llm, ctx.pubmed, verify=VerifyConfig()).answer(item.question)


async def ranking_agent(item: EvalItem, ctx: Context) -> CitedAnswer:
    """E: C with the screening call replaced by embedding similarity. Plan, search once, keep the ten
    abstracts closest to the question, answer like B and C (src/bioscout/agent/rank.py)."""
    ranker = ctx.ranker or default_ranker()
    return await ResearchAgent(ctx.llm, ctx.pubmed, ranker=ranker).answer(item.question)


VARIANTS: dict[str, Variant] = {
    "A": Variant("A", "LLM only", SYSTEM_LLM_ONLY, llm_only),
    "B": Variant("B", "Naive RAG", SYSTEM_GROUNDED, naive_rag),
    "C": Variant("C", "Agent", SYSTEM_GROUNDED, agent, ResearchAgent.fingerprint()),
    "D": Variant(
        "D",
        "Agent + verifier",
        SYSTEM_GROUNDED,
        verified_agent,
        (*ResearchAgent.fingerprint(), *verify_fingerprint()),
    ),
    "E": Variant(
        "E",
        "Agent, embedding rank",
        SYSTEM_GROUNDED,
        ranking_agent,
        ResearchAgent.rank_fingerprint(f"embedding:{EMBED_MODEL}"),
    ),
}
