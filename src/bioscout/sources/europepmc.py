"""Open-access full text via Europe PMC REST.

Docs: https://europepmc.org/RestfulWebService
Only open-access articles return full text; we read them live and never store them in the repo.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET

from bioscout.http import CachedHttp
from bioscout.models import FullText, FullTextSection

EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"


def _clean(text: str) -> str:
    return " ".join(text.split())


class EuropePMC:
    def __init__(self, http: CachedHttp) -> None:
        self.http = http

    async def resolve_pmcid(self, pmid: str) -> tuple[str | None, bool]:
        """Map a PMID to its PMCID and open-access flag."""
        params = {"query": f"EXT_ID:{pmid} AND SRC:MED", "format": "json", "resultType": "lite"}
        data = json.loads(await self.http.get_text(f"{EPMC}/search", params))
        results = data.get("resultList", {}).get("result", [])
        if not results:
            return None, False
        hit = results[0]
        return hit.get("pmcid"), hit.get("isOpenAccess") == "Y"

    async def fulltext(
        self, pmid: str | None = None, pmcid: str | None = None, max_chars: int = 12000
    ) -> FullText:
        if not pmid and not pmcid:
            return FullText(available=False, reason="Provide a pmid or a pmcid.")
        if not pmcid and pmid:
            pmcid, is_oa = await self.resolve_pmcid(pmid)
            if not pmcid:
                return FullText(pmid=pmid, available=False, reason="No PMC record: use the abstract instead.")
            if not is_oa:
                return FullText(
                    pmid=pmid,
                    pmcid=pmcid,
                    available=False,
                    reason="Not open access: use the abstract instead.",
                )
        pmcid = pmcid.upper() if pmcid.upper().startswith("PMC") else f"PMC{pmcid}"
        xml = await self.http.get_text(f"{EPMC}/{pmcid}/fullTextXML")
        sections, truncated = parse_jats_sections(xml, max_chars)
        if not sections:
            return FullText(pmid=pmid, pmcid=pmcid, available=False, reason="Full text has no body sections.")
        return FullText(pmid=pmid, pmcid=pmcid, available=True, sections=sections, truncated=truncated)


def parse_jats_sections(xml: str, max_chars: int = 12000) -> tuple[list[FullTextSection], bool]:
    """Top-level <sec> blocks of a JATS <body>, each as title + paragraph text, within a char budget."""
    root = ET.fromstring(xml)
    body = root.find(".//body")
    if body is None:
        return [], False
    sections: list[FullTextSection] = []
    used = 0
    for sec in body.findall("sec"):
        title = _clean("".join(sec.find("title").itertext())) if sec.find("title") is not None else "Untitled"
        paragraphs = [_clean("".join(p.itertext())) for p in sec.iter("p")]
        text = "\n".join(p for p in paragraphs if p)
        if not text:
            continue
        room = max_chars - used
        if room <= 0:
            return sections, True
        if len(text) > room:
            sections.append(FullTextSection(title=title, text=text[:room].rstrip() + " …"))
            return sections, True
        sections.append(FullTextSection(title=title, text=text))
        used += len(text)
    return sections, False
