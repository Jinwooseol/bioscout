"""Answer variants the eval compares. All use the same model, effort and output schema, so differences
in the results come from the design (retrieval, agent, verifier), not from the model."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from bioscout.agent import ResearchAgent
from bioscout.answer import CitedAnswer, SearchLog, Usage
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

SYSTEM_LLM_ONLY = f"""\
You are a biomedical research assistant answering from your knowledge of the published literature.
{ANSWER_RULES}
Cite only papers you are confident exist and say what the sentence claims. If you are unsure of a \
PMID, leave "citations" empty rather than guess."""


@dataclass
class Context:
    llm: LLM
    pubmed: Literature


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
    result = await ctx.llm.json(
        system=SYSTEM_LLM_ONLY, user=f"Question: {item.question}", schema=ANSWER_SCHEMA
    )
    return answer_from_result(item.question, result, usage=result.usage)


async def naive_rag(item: EvalItem, ctx: Context) -> CitedAnswer:
    """B: the question verbatim as a PubMed query, top 10 abstracts as context, one LLM call."""
    found = await ctx.pubmed.search(item.question, max_results=10)
    search = SearchLog(
        query=item.question, pmids=[a.pmid for a in found.articles], total_found=found.total_found
    )
    batch = await ctx.pubmed.abstracts(search.pmids) if search.pmids else AbstractBatch(articles=[])
    context = format_abstracts(batch.articles) or "(The PubMed search returned no articles.)"
    user = (
        f"Question: {item.question}\n\n"
        f"Abstracts from a PubMed search for the question, most relevant first:\n\n{context}"
    )
    result = await ctx.llm.json(system=SYSTEM_GROUNDED, user=user, schema=ANSWER_SCHEMA)
    tools = Usage(tool_calls=1 + bool(search.pmids))
    return answer_from_result(
        item.question,
        result,
        usage=Usage.total([result.usage, tools]),
        searches=[search],
        context_pmids=[a.pmid for a in batch.articles],
    )


async def agent(item: EvalItem, ctx: Context) -> CitedAnswer:
    """C: plan, search, screen (and search once more if needed), then answer like B (src/bioscout/agent)."""
    return await ResearchAgent(ctx.llm, ctx.pubmed).answer(item.question)


VARIANTS: dict[str, Variant] = {
    "A": Variant("A", "LLM only", SYSTEM_LLM_ONLY, llm_only),
    "B": Variant("B", "Naive RAG", SYSTEM_GROUNDED, naive_rag),
    "C": Variant("C", "Agent", SYSTEM_GROUNDED, agent, ResearchAgent.fingerprint()),
}
