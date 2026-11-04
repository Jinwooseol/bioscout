"""Answer variants the eval compares. All use the same model, effort and output schema, so differences
in the results come from the design (retrieval, agent, verifier), not from the model."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol

from bioscout.answer import CitedAnswer, CitedSentence, SearchLog
from bioscout.evals.datasets import EvalItem
from bioscout.llm import LLM, LLMResult
from bioscout.models import AbstractBatch, PmidCheck, SearchResult

ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["yes", "no", "maybe", "not_applicable"]},
        "sentences": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "citations": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["text", "citations"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["decision", "sentences"],
    "additionalProperties": False,
}

_RULES = """\
Write 2 to 6 sentences in English. Each sentence states one point and lists in "citations" the PubMed \
IDs (PMIDs, digits only) of the papers that support it.
Set "decision" to yes, no or maybe when the question can be answered that way, otherwise not_applicable.
The answer supports reading the research literature; it is not advice about an individual patient."""

SYSTEM_LLM_ONLY = f"""\
You are a biomedical research assistant answering from your knowledge of the published literature.
{_RULES}
Cite only papers you are confident exist and say what the sentence claims. If you are unsure of a \
PMID, leave "citations" empty rather than guess."""

SYSTEM_NAIVE_RAG = f"""\
You are a biomedical research assistant. Answer using only the PubMed abstracts given with the question.
{_RULES}
Cite only PMIDs of the given abstracts, and only where the abstract supports the sentence. If the \
abstracts do not answer the question, say so and leave "citations" empty."""


class Literature(Protocol):
    """The slice of the PubMed client that variants and scoring use."""

    async def search(self, query: str, max_results: int = 10) -> SearchResult: ...

    async def abstracts(self, pmids: list[str]) -> AbstractBatch: ...

    async def check_pmids(self, pmids: list[str]) -> PmidCheck: ...


@dataclass
class Context:
    llm: LLM
    pubmed: Literature


@dataclass(frozen=True)
class Variant:
    key: str
    name: str
    system_prompt: str
    run: Callable[[EvalItem, Context], Awaitable[CitedAnswer]]

    @property
    def prompt_sha(self) -> str:
        """Changes whenever the prompt or schema changes, so runs record exactly what was asked."""
        raw = json.dumps([self.system_prompt, ANSWER_SCHEMA], sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def _to_answer(item: EvalItem, result: LLMResult, searches: list[SearchLog], tool_calls: int) -> CitedAnswer:
    usage = result.usage.model_copy()
    usage.tool_calls += tool_calls
    if result.refused or result.data is None:
        return CitedAnswer(question=item.question, searches=searches, usage=usage, refused=True)
    sentences = [
        CitedSentence(text=s["text"].strip(), citations=[str(c).strip() for c in s["citations"]])
        for s in result.data["sentences"]
    ]
    return CitedAnswer(
        question=item.question,
        decision=result.data["decision"],
        sentences=sentences,
        searches=searches,
        usage=usage,
    )


async def llm_only(item: EvalItem, ctx: Context) -> CitedAnswer:
    """A: no tools. Measures how often the model invents or misattributes citations on its own."""
    result = await ctx.llm.json(
        system=SYSTEM_LLM_ONLY, user=f"Question: {item.question}", schema=ANSWER_SCHEMA
    )
    return _to_answer(item, result, searches=[], tool_calls=0)


async def naive_rag(item: EvalItem, ctx: Context) -> CitedAnswer:
    """B: the question verbatim as a PubMed query, top 10 abstracts as context, one LLM call."""
    found = await ctx.pubmed.search(item.question, max_results=10)
    search = SearchLog(
        query=item.question, pmids=[a.pmid for a in found.articles], total_found=found.total_found
    )
    batch = await ctx.pubmed.abstracts(search.pmids) if search.pmids else AbstractBatch(articles=[])
    if batch.articles:
        context = "\n\n".join(
            f"PMID {a.pmid} | {a.title} | {a.journal} {a.year or ''}\n{a.abstract_text or '(no abstract)'}"
            for a in batch.articles
        )
    else:
        context = "(The PubMed search returned no articles.)"
    user = (
        f"Question: {item.question}\n\n"
        f"Abstracts from a PubMed search for the question, most relevant first:\n\n{context}"
    )
    result = await ctx.llm.json(system=SYSTEM_NAIVE_RAG, user=user, schema=ANSWER_SCHEMA)
    tool_calls = 1 + bool(search.pmids)
    return _to_answer(item, result, searches=[search], tool_calls=tool_calls)


VARIANTS: dict[str, Variant] = {
    "A": Variant("A", "LLM only", SYSTEM_LLM_ONLY, llm_only),
    "B": Variant("B", "Naive RAG", SYSTEM_NAIVE_RAG, naive_rag),
}
