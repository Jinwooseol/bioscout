"""The agent on the MCP server's tools: same answers as in-process, and tool errors that are not swallowed."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from bioscout.agent import ResearchAgent
from bioscout.config import Settings
from bioscout.models import SearchResult
from bioscout.tools import McpLiterature, ToolCallError, open_literature
from tests.fakes import FakeLLM, FakePubMed, reply
from tests.test_agent import plan, screen

TEXTS = {"10000001": "Drug X lowered mortality.", "10000002": "Drug X caused nausea."}
PLAN = plan("drug X mortality")
SCREEN = screen(["10000001", "10000002"])
ANSWER = reply("yes", ("X lowered mortality.", ["10000001"]))


class ToolSidePubMed(FakePubMed):
    """The fake PubMed behind the server: the search tool also passes sort and year filters."""

    async def search(self, query: str, max_results: int = 10, *_: Any) -> SearchResult:
        return await super().search(query, max_results)


def sources(pubmed: FakePubMed) -> Any:
    return SimpleNamespace(pubmed=pubmed)


async def test_the_agent_gives_the_same_answer_on_the_mcp_tools_as_in_process():
    direct = await ResearchAgent(FakeLLM([PLAN, SCREEN, ANSWER]), FakePubMed(texts=TEXTS)).answer(
        "Is X good?"
    )

    behind_server = ToolSidePubMed(texts=TEXTS)
    async with open_literature(sources(behind_server), through_mcp=True) as tools:
        assert isinstance(tools, McpLiterature)
        over_mcp = await ResearchAgent(FakeLLM([PLAN, SCREEN, ANSWER]), tools).answer("Is X good?")

    assert over_mcp.sentences == direct.sentences and over_mcp.context_pmids == direct.context_pmids
    assert [search.model_dump() for search in over_mcp.searches] == [s.model_dump() for s in direct.searches]
    assert behind_server.queries == ["Is X good?", "drug X mortality"]  # the server did the searching


async def test_long_lists_are_split_at_the_limits_the_tools_declare():
    behind_server = ToolSidePubMed()
    pmids = [str(10000000 + n) for n in range(120)]
    async with open_literature(sources(behind_server), through_mcp=True) as tools:
        batch = await tools.abstracts(pmids)
        check = await tools.check_pmids([*pmids, *pmids])
        assert (await tools.abstracts([])).articles == []  # nothing to fetch, so no call at all
    assert [len(call) for call in behind_server.fetched] == [50, 50, 20]
    assert [article.pmid for article in batch.articles] == pmids
    assert len(check.existing) + len(check.missing) == 240


async def test_a_tool_error_is_raised_with_the_tools_own_message():
    async with open_literature(sources(ToolSidePubMed()), through_mcp=True) as tools:
        with pytest.raises(ToolCallError, match="search_pubmed"):
            await tools.search("MSI", max_results=500)  # the tool accepts at most 50


async def test_without_the_switch_the_client_is_used_directly():
    pubmed = FakePubMed()
    async with open_literature(sources(pubmed)) as tools:
        assert tools is pubmed
    assert Settings().tools == "direct" and not Settings().through_mcp
    assert Settings(tools="mcp").through_mcp
    with pytest.raises(ValueError, match="BIOSCOUT_TOOLS"):
        Settings(tools="other")
