"""Typed records returned by the data sources and exposed as MCP structured output."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ArticleSummary(BaseModel):
    pmid: str
    title: str
    journal: str = ""
    year: int | None = None
    first_author: str = ""
    doi: str | None = None
    pmcid: str | None = None

    @property
    def url(self) -> str:
        return f"https://pubmed.ncbi.nlm.nih.gov/{self.pmid}/"


class SearchResult(BaseModel):
    query: str
    total_found: int = Field(description="Total PubMed hits for the query, not just those returned")
    articles: list[ArticleSummary]


class AbstractSection(BaseModel):
    label: str | None = None
    text: str


class ArticleAbstract(BaseModel):
    pmid: str
    title: str
    journal: str = ""
    year: int | None = None
    abstract: list[AbstractSection] = Field(default_factory=list)
    mesh_terms: list[str] = Field(default_factory=list)
    publication_types: list[str] = Field(default_factory=list)
    doi: str | None = None
    pmcid: str | None = None

    @property
    def abstract_text(self) -> str:
        return "\n".join(f"{s.label}: {s.text}" if s.label else s.text for s in self.abstract)


class AbstractBatch(BaseModel):
    articles: list[ArticleAbstract]
    not_found: list[str] = Field(default_factory=list)


class PmidCheck(BaseModel):
    existing: list[str]
    missing: list[str] = Field(description="PMIDs that PubMed does not know: likely hallucinated")


class FullTextSection(BaseModel):
    title: str
    text: str


class FullText(BaseModel):
    pmid: str | None = None
    pmcid: str | None = None
    available: bool
    reason: str | None = Field(default=None, description="Why full text is unavailable, if it is")
    sections: list[FullTextSection] = Field(default_factory=list)
    truncated: bool = False


class TrialSummary(BaseModel):
    nct_id: str
    title: str
    status: str = ""
    phases: list[str] = Field(default_factory=list)
    conditions: list[str] = Field(default_factory=list)
    interventions: list[str] = Field(default_factory=list)
    enrollment: int | None = None
    start_date: str | None = None

    @property
    def url(self) -> str:
        return f"https://clinicaltrials.gov/study/{self.nct_id}"


class TrialSearchResult(BaseModel):
    query: str
    trials: list[TrialSummary]
