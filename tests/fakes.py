"""Offline stand-ins shared by the eval and agent tests: a scripted LLM and an in-memory PubMed whose
PMIDs are synthetic (10000001...)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from bioscout.answer import Usage
from bioscout.llm import LLMResult
from bioscout.models import AbstractBatch, ArticleAbstract, ArticleSummary, PmidCheck, SearchResult

REAL = {"10000001", "10000002", "10000003"}  # the only PMIDs the fake PubMed says exist


@dataclass
class FakeLLM:
    replies: list[dict[str, Any] | None]  # answered in order; None = refusal
    model: str = "claude-opus-5-5"
    effort: str = "medium"
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def json(self, *, system: str, user: str, schema: dict, max_tokens: int = 16000) -> LLMResult:
        self.calls.append({"system": system, "user": user, "schema": schema})
        data = self.replies.pop(0)
        usage = Usage(usd=0.01, llm_calls=1, input_tokens=100, output_tokens=50, model=self.model)
        return LLMResult(data=data, usage=usage, stop_reason="refusal" if data is None else "end_turn")


class FakePubMed:
    def __init__(self, hits: list[str] | None = None, by_query: dict[str, list[str]] | None = None) -> None:
        self.hits = hits if hits is not None else ["10000002", "10000001"]  # for any other query
        self.by_query = by_query or {}
        self.queries: list[str] = []
        self.fetched: list[list[str]] = []  # one entry per abstracts() call

    async def search(self, query: str, max_results: int = 10) -> SearchResult:
        self.queries.append(query)
        hits = self.by_query.get(query, self.hits)
        arts = [ArticleSummary(pmid=p, title=f"Synthetic {p}") for p in hits[:max_results]]
        return SearchResult(query=query, total_found=len(hits), articles=arts)

    async def abstracts(self, pmids: list[str]) -> AbstractBatch:
        self.fetched.append(list(pmids))
        return AbstractBatch(articles=[ArticleAbstract(pmid=p, title=f"Synthetic {p}") for p in pmids])

    async def check_pmids(self, pmids: list[str]) -> PmidCheck:
        return PmidCheck(
            existing=[p for p in pmids if p in REAL], missing=[p for p in pmids if p not in REAL]
        )


def reply(decision: str, *sentences: tuple[str, list[str]]) -> dict[str, Any]:
    """An ANSWER_SCHEMA reply."""
    return {"decision": decision, "sentences": [{"text": t, "citations": c} for t, c in sentences]}
