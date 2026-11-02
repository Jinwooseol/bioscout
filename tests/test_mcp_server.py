"""End-to-end through the MCP protocol, in-process, with mocked upstream APIs."""

from __future__ import annotations

from mcp import Client

from bioscout.mcp_server import Sources, create_server
from bioscout.sources.pubmed import EUTILS
from tests.conftest import fixture, ok

EXPECTED_TOOLS = {"search_pubmed", "get_abstracts", "check_pmids", "get_fulltext", "search_trials"}


async def test_lists_five_read_only_tools(settings):
    async with Client(create_server(Sources(settings))) as client:
        tools = (await client.list_tools()).tools
    assert {t.name for t in tools} == EXPECTED_TOOLS
    assert all(t.annotations and t.annotations.read_only_hint for t in tools)


async def test_search_pubmed_returns_structured_content(settings, api):
    api.get(f"{EUTILS}/esearch.fcgi").mock(return_value=ok(fixture("esearch.json")))
    api.get(f"{EUTILS}/esummary.fcgi").mock(return_value=ok(fixture("esummary.json")))

    async with Client(create_server(Sources(settings))) as client:
        result = await client.call_tool("search_pubmed", {"query": "MSI", "max_results": 2})

    assert not result.is_error
    pmids = [a["pmid"] for a in result.structured_content["articles"]]
    assert pmids == ["10000002", "10000001"]


async def test_check_pmids_tool(settings, api):
    api.get(f"{EUTILS}/esummary.fcgi").mock(return_value=ok(fixture("esummary.json")))
    async with Client(create_server(Sources(settings))) as client:
        result = await client.call_tool("check_pmids", {"pmids": ["10000002", "99999999"]})
    assert result.structured_content["missing"] == ["99999999"]


async def test_invalid_arguments_are_reported_not_raised(settings):
    async with Client(create_server(Sources(settings))) as client:
        result = await client.call_tool("search_pubmed", {"query": "MSI", "max_results": 500})
    assert result.is_error


async def test_upstream_outage_becomes_a_readable_tool_error(settings, api, monkeypatch):
    import httpx

    async def no_sleep(_s: float) -> None:
        return None

    monkeypatch.setattr("asyncio.sleep", no_sleep)
    api.get(f"{EUTILS}/esearch.fcgi").mock(side_effect=httpx.ConnectError("down"))
    async with Client(create_server(Sources(settings))) as client:
        result = await client.call_tool("search_pubmed", {"query": "MSI"})
    assert result.is_error
    assert "temporarily unreachable" in result.content[0].text
