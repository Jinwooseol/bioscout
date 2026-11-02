"""Bio MCP server: PubMed, Europe PMC and ClinicalTrials.gov as MCP tools.

Run over stdio (what Claude Desktop / Claude Code expect):

    uv run bioscout-mcp
"""

from __future__ import annotations

import functools
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import cached_property
from typing import Annotated, ParamSpec, TypeVar

import httpx
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from bioscout.config import Settings
from bioscout.http import CachedHttp, CacheMissError, UpstreamError
from bioscout.models import (
    AbstractBatch,
    FullText,
    PmidCheck,
    SearchResult,
    TrialSearchResult,
)
from bioscout.sources import ClinicalTrials, EuropePMC, PubMed

INSTRUCTIONS = """\
Biomedical literature tools. Typical flow: search_pubmed -> get_abstracts on the most relevant
PMIDs -> cite PMIDs for every claim. Use check_pmids before presenting any PMID you did not get
from these tools. get_fulltext works only for open-access papers; fall back to the abstract.
These tools support research reading, not clinical decisions about individual patients."""

P = ParamSpec("P")
R = TypeVar("R")


def agent_readable_errors(fn: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
    """Turn upstream failures into short tool errors an agent can act on, instead of tracebacks."""

    @functools.wraps(fn)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return await fn(*args, **kwargs)
        except UpstreamError as exc:
            raise ToolError(
                f"Source temporarily unreachable, retry later or use another tool. ({exc})"
            ) from exc
        except CacheMissError as exc:
            raise ToolError(f"Replay mode: no recorded response for this call. ({exc})") from exc
        except httpx.HTTPStatusError as exc:
            raise ToolError(
                f"Source rejected the request (HTTP {exc.response.status_code}); check the arguments."
            ) from exc

    return wrapper


READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=True, idempotent_hint=True)


@dataclass
class Sources:
    settings: Settings

    @cached_property
    def http(self) -> CachedHttp:
        return CachedHttp(
            self.settings,
            rate_limits={
                "eutils.ncbi.nlm.nih.gov": self.settings.ncbi_requests_per_second,
                "www.ebi.ac.uk": 5.0,
                "clinicaltrials.gov": 5.0,
            },
        )

    @cached_property
    def pubmed(self) -> PubMed:
        return PubMed(self.http)

    @cached_property
    def europepmc(self) -> EuropePMC:
        return EuropePMC(self.http)

    @cached_property
    def trials(self) -> ClinicalTrials:
        return ClinicalTrials(self.http)


def create_server(sources: Sources | None = None) -> MCPServer:
    src = sources or Sources(Settings.from_env())
    server = MCPServer("bioscout", title="BioScout biomedical literature tools", instructions=INSTRUCTIONS)

    @server.tool(annotations=READ_ONLY)
    @agent_readable_errors
    async def search_pubmed(
        query: Annotated[
            str, Field(description="PubMed query. Boolean operators, [MeSH] and [tiab] tags work.")
        ],
        max_results: Annotated[int, Field(ge=1, le=50)] = 10,
        min_year: int | None = None,
        max_year: int | None = None,
        sort: Annotated[str, Field(pattern="^(relevance|pub_date)$")] = "relevance",
    ) -> SearchResult:
        """Search PubMed. Returns PMIDs with title, journal, year and first author."""
        return await src.pubmed.search(query, max_results, sort, min_year, max_year)

    @server.tool(annotations=READ_ONLY)
    @agent_readable_errors
    async def get_abstracts(
        pmids: Annotated[list[str], Field(min_length=1, max_length=50)],
    ) -> AbstractBatch:
        """Fetch abstracts (with section labels), MeSH terms and publication types for PMIDs."""
        return await src.pubmed.abstracts(pmids)

    @server.tool(annotations=READ_ONLY)
    @agent_readable_errors
    async def check_pmids(
        pmids: Annotated[list[str], Field(min_length=1, max_length=200)],
    ) -> PmidCheck:
        """Check which PMIDs exist in PubMed. Missing ones are likely fabricated citations."""
        return await src.pubmed.check_pmids(pmids)

    @server.tool(annotations=READ_ONLY)
    @agent_readable_errors
    async def get_fulltext(
        pmid: str | None = None,
        pmcid: str | None = None,
        max_chars: Annotated[int, Field(ge=1000, le=50000)] = 12000,
    ) -> FullText:
        """Open-access full text from Europe PMC, split into sections. Give a pmid or a pmcid."""
        return await src.europepmc.fulltext(pmid=pmid, pmcid=pmcid, max_chars=max_chars)

    @server.tool(annotations=READ_ONLY)
    @agent_readable_errors
    async def search_trials(
        query: str | None = None,
        condition: str | None = None,
        intervention: str | None = None,
        status: Annotated[
            list[str] | None,
            Field(description="e.g. RECRUITING, COMPLETED, ACTIVE_NOT_RECRUITING"),
        ] = None,
        max_results: Annotated[int, Field(ge=1, le=50)] = 10,
    ) -> TrialSearchResult:
        """Search ClinicalTrials.gov by free text, condition and/or intervention."""
        return await src.trials.search(query, condition, intervention, status, max_results)

    return server


def main() -> None:
    create_server().run("stdio")


if __name__ == "__main__":
    main()
