"""The agent's PubMed tools, reached in-process or through the MCP server.

The MCP server offers search_pubmed, get_abstracts and check_pmids to any MCP host. `McpLiterature`
gives the agent the same three calls through an MCP client session, so the agent can run on exactly
the tools a host such as Claude Code gets, and a run exercises the tool layer (its schemas, limits and
error messages) and not only the code behind it. BIOSCOUT_TOOLS=mcp turns it on.

Both ways end in the same PubMed client and the same response cache, so they return the same data.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from mcp import Client
from pydantic import BaseModel

from bioscout.mcp_server import Sources, create_server
from bioscout.models import AbstractBatch, PmidCheck, SearchResult
from bioscout.sources.pubmed import Literature

# The most PMIDs the tools accept in one call.
ABSTRACTS_PER_CALL = 50
PMIDS_PER_CHECK = 200


class ToolCallError(RuntimeError):
    """An MCP tool reported an error: arguments it does not accept, or an upstream outage."""


class McpLiterature:
    """`Literature` over an MCP client session."""

    def __init__(self, client: Client) -> None:
        self.client = client

    async def _call[T: BaseModel](self, tool: str, model: type[T], **arguments: Any) -> T:
        result = await self.client.call_tool(tool, arguments)
        if result.is_error:
            detail = " ".join(getattr(part, "text", "") for part in result.content).strip()
            raise ToolCallError(f"{tool}: {detail}")
        return model.model_validate(result.structured_content)

    async def search(self, query: str, max_results: int = 10) -> SearchResult:
        return await self._call("search_pubmed", SearchResult, query=query, max_results=max_results)

    async def abstracts(self, pmids: list[str]) -> AbstractBatch:
        batch = AbstractBatch(articles=[])
        for start in range(0, len(pmids), ABSTRACTS_PER_CALL):
            chunk = pmids[start : start + ABSTRACTS_PER_CALL]
            part = await self._call("get_abstracts", AbstractBatch, pmids=chunk)
            batch.articles += part.articles
            batch.not_found += part.not_found
        return batch

    async def check_pmids(self, pmids: list[str]) -> PmidCheck:
        check = PmidCheck(existing=[], missing=[])
        for start in range(0, len(pmids), PMIDS_PER_CHECK):
            part = await self._call("check_pmids", PmidCheck, pmids=pmids[start : start + PMIDS_PER_CHECK])
            check.existing += part.existing
            check.missing += part.missing
        return check


@asynccontextmanager
async def open_literature(sources: Sources, through_mcp: bool = False) -> AsyncIterator[Literature]:
    """The PubMed tools for one run: the client itself, or the MCP server's tools over a client session."""
    if not through_mcp:
        yield sources.pubmed
        return
    async with Client(create_server(sources)) as client:
        yield McpLiterature(client)
