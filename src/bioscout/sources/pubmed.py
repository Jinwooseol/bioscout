"""PubMed via NCBI E-utilities (esearch, esummary, efetch).

Docs: https://www.ncbi.nlm.nih.gov/books/NBK25501/
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from typing import Any, Protocol

from bioscout.http import CachedHttp
from bioscout.models import (
    AbstractBatch,
    AbstractSection,
    ArticleAbstract,
    ArticleSummary,
    PmidCheck,
    SearchResult,
)

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_PMID_RE = re.compile(r"^\d{1,9}$")


def _year(text: str | None) -> int | None:
    match = re.search(r"(19|20)\d{2}", text or "")
    return int(match.group()) if match else None


def normalize_pmids(pmids: list[str]) -> list[str]:
    """Strip whitespace and 'PMID:' prefixes, drop malformed ids, keep order, dedupe."""
    seen: dict[str, None] = {}
    for raw in pmids:
        pmid = str(raw).strip().removeprefix("PMID:").strip()
        if _PMID_RE.match(pmid):
            seen.setdefault(pmid, None)
    return list(seen)


class Literature(Protocol):
    """The slice of PubMed that the agent, the eval variants and scoring use (faked in tests)."""

    async def search(self, query: str, max_results: int = 10) -> SearchResult: ...

    async def abstracts(self, pmids: list[str]) -> AbstractBatch: ...

    async def check_pmids(self, pmids: list[str]) -> PmidCheck: ...


class PubMed:
    def __init__(self, http: CachedHttp) -> None:
        self.http = http

    def _params(self, **params: Any) -> dict[str, Any]:
        s = self.http.settings
        base: dict[str, Any] = {"db": "pubmed", "tool": s.ncbi_tool}
        if s.ncbi_email:
            base["email"] = s.ncbi_email
        if s.ncbi_api_key:
            base["api_key"] = s.ncbi_api_key
        return base | params

    async def search(
        self,
        query: str,
        max_results: int = 10,
        sort: str = "relevance",
        min_year: int | None = None,
        max_year: int | None = None,
    ) -> SearchResult:
        params = self._params(term=query, retmax=max(1, min(max_results, 50)), retmode="json", sort=sort)
        if min_year or max_year:
            params |= {"datetype": "pdat", "mindate": min_year or 1900, "maxdate": max_year or 3000}
        data = json.loads(await self.http.get_text(f"{EUTILS}/esearch.fcgi", params))
        result = data.get("esearchresult", {})
        ids = result.get("idlist", [])
        summaries = await self.summaries(ids) if ids else {}
        return SearchResult(
            query=query,
            total_found=int(result.get("count", 0)),
            articles=[summaries[i] for i in ids if i in summaries],
        )

    async def summaries(self, pmids: list[str]) -> dict[str, ArticleSummary]:
        """esummary for up to 200 PMIDs. Unknown PMIDs are simply absent from the result."""
        ids = normalize_pmids(pmids)
        if not ids:
            return {}
        params = self._params(id=",".join(ids), retmode="json")
        data = json.loads(await self.http.get_text(f"{EUTILS}/esummary.fcgi", params))
        result = data.get("result", {})
        out: dict[str, ArticleSummary] = {}
        for uid in result.get("uids", []):
            doc = result.get(uid) or {}
            if "error" in doc or not doc.get("title"):
                continue
            article_ids = {a.get("idtype"): a.get("value") for a in doc.get("articleids", [])}
            authors = [a.get("name", "") for a in doc.get("authors", []) if a.get("authtype") == "Author"]
            out[uid] = ArticleSummary(
                pmid=uid,
                title=doc["title"].strip(),
                journal=doc.get("source", ""),
                year=_year(doc.get("pubdate")),
                first_author=authors[0] if authors else "",
                doi=article_ids.get("doi"),
                pmcid=article_ids.get("pmc"),
            )
        return out

    async def check_pmids(self, pmids: list[str]) -> PmidCheck:
        """Which of these PMIDs exist? The basis of the 'hallucinated citation rate' metric."""
        ids = normalize_pmids(pmids)
        found = await self.summaries(ids) if ids else {}
        malformed = [p for p in pmids if str(p).strip().removeprefix("PMID:").strip() not in ids]
        return PmidCheck(
            existing=[p for p in ids if p in found],
            missing=[p for p in ids if p not in found] + malformed,
        )

    async def abstracts(self, pmids: list[str]) -> AbstractBatch:
        ids = normalize_pmids(pmids)[:50]
        if not ids:
            return AbstractBatch(articles=[], not_found=list(pmids))
        params = self._params(id=",".join(ids), retmode="xml", rettype="abstract")
        xml = await self.http.get_text(f"{EUTILS}/efetch.fcgi", params)
        articles = parse_pubmed_xml(xml)
        got = {a.pmid for a in articles}
        return AbstractBatch(articles=articles, not_found=[p for p in ids if p not in got])


def _text(el: ET.Element | None) -> str:
    return " ".join("".join(el.itertext()).split()) if el is not None else ""


def parse_pubmed_xml(xml: str) -> list[ArticleAbstract]:
    root = ET.fromstring(xml)
    out: list[ArticleAbstract] = []
    for art in root.iter("PubmedArticle"):
        cit = art.find("MedlineCitation")
        if cit is None:
            continue
        article = cit.find("Article")
        pmid = _text(cit.find("PMID"))
        sections = [
            AbstractSection(label=t.get("Label") or None, text=_text(t))
            for t in (article.findall("Abstract/AbstractText") if article is not None else [])
            if _text(t)
        ]
        ids = {i.get("IdType"): _text(i) for i in art.findall("PubmedData/ArticleIdList/ArticleId")}
        pub_date = article.find("Journal/JournalIssue/PubDate") if article is not None else None
        year = _year(_text(pub_date.find("Year")) or _text(pub_date)) if pub_date is not None else None
        out.append(
            ArticleAbstract(
                pmid=pmid,
                title=_text(article.find("ArticleTitle")) if article is not None else "",
                journal=_text(article.find("Journal/Title")) if article is not None else "",
                year=year,
                abstract=sections,
                mesh_terms=[_text(d) for d in cit.findall("MeshHeadingList/MeshHeading/DescriptorName")],
                publication_types=[
                    _text(p)
                    for p in (
                        article.findall("PublicationTypeList/PublicationType") if article is not None else []
                    )
                ],
                doi=ids.get("doi"),
                pmcid=ids.get("pmc"),
            )
        )
    return out
