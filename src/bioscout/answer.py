"""The answer data contract. Variants, the demo and the eval harness all produce or read CitedAnswer."""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

Decision = Literal["yes", "no", "maybe", "not_applicable"]


class Verdict(StrEnum):
    SUPPORTED = "supported"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"


class CitedSentence(BaseModel):
    text: str
    citations: list[str] = Field(default_factory=list, description="PMIDs that support this sentence")
    verdict: Verdict | None = Field(default=None, description="Set by the verifier (Day 4)")
    evidence: str | None = Field(default=None, description="Abstract sentence the verdict rests on")


class DroppedSentence(BaseModel):
    text: str
    citations: list[str] = Field(default_factory=list)
    reason: str


class SearchLog(BaseModel):
    query: str
    pmids: list[str] = Field(default_factory=list, description="Returned PMIDs, in rank order")
    total_found: int = 0


class Usage(BaseModel):
    usd: float | None = Field(default=0.0, description="None when a model's price is unknown")
    latency_ms: int = 0
    tool_calls: int = 0
    llm_calls: int = 0
    cached_llm_calls: int = Field(default=0, description="LLM responses replayed from the cache")
    input_tokens: int = 0
    output_tokens: int = 0
    model: str | None = Field(default=None, description="Model that produced the final answer")
    fell_back: bool = Field(default=False, description="A fallback model answered after a refusal")

    @classmethod
    def total(cls, usages: Iterable[Usage]) -> Usage:
        out = cls()
        for usage in usages:
            out.add(usage)
        return out

    def add(self, other: Usage) -> None:
        """Accumulate another call's usage. Latency is measured end to end by the caller instead."""
        self.usd = None if self.usd is None or other.usd is None else self.usd + other.usd
        self.tool_calls += other.tool_calls
        self.llm_calls += other.llm_calls
        self.cached_llm_calls += other.cached_llm_calls
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.model = other.model or self.model
        self.fell_back = self.fell_back or other.fell_back


class CitedAnswer(BaseModel):
    question: str
    decision: Decision | None = None
    sentences: list[CitedSentence] = Field(default_factory=list)
    dropped: list[DroppedSentence] = Field(default_factory=list)
    searches: list[SearchLog] = Field(default_factory=list)
    context_pmids: list[str] = Field(
        default_factory=list, description="PMIDs whose abstracts the answering step was given"
    )
    usage: Usage = Field(default_factory=Usage)
    refused: bool = False

    @property
    def cited_pmids(self) -> list[str]:
        """Unique cited PMIDs in first-cited order."""
        seen: dict[str, None] = {}
        for s in self.sentences:
            for pmid in s.citations:
                seen.setdefault(pmid.strip(), None)
        return list(seen)

    @property
    def retrieved_pmids(self) -> list[str]:
        """Unique PMIDs returned by any search, in first-seen order."""
        seen: dict[str, None] = {}
        for search in self.searches:
            for pmid in search.pmids:
                seen.setdefault(pmid, None)
        return list(seen)
