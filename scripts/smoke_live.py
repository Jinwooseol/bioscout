"""Live smoke test: start the MCP server over stdio and call every tool against the real APIs.

    uv run python scripts/smoke_live.py            # all five tools, real network
    uv run python scripts/smoke_live.py --offline  # only start the server and list its tools

Exit code 0 means every check passed.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

from mcp import Client, StdioServerParameters

SERVER = StdioServerParameters(command=sys.executable, args=["-m", "bioscout.mcp_server"])

CHECKS: list[tuple[str, dict, str]] = [
    (
        "search_pubmed",
        {"query": "microsatellite instability[tiab] AND digital PCR", "max_results": 5},
        "articles",
    ),
    ("get_abstracts", {"pmids": []}, "articles"),  # PMIDs filled from the search
    ("check_pmids", {"pmids": ["99999999999"]}, "missing"),  # filled with search PMIDs + a fake one
    ("get_fulltext", {}, "sections"),  # filled with the first open-access hit
    ("search_trials", {"condition": "colorectal cancer", "query": "MSI-H", "max_results": 3}, "trials"),
]


async def run(offline: bool) -> int:
    failures = 0
    async with Client(SERVER) as client:
        tools = sorted(t.name for t in (await client.list_tools()).tools)
        print(f"server up, {len(tools)} tools: {', '.join(tools)}")
        if offline:
            return 0 if len(tools) == 5 else 1

        pmids: list[str] = []
        pmcid: str | None = None
        for name, args, key in CHECKS:
            if name in {"get_abstracts", "check_pmids", "get_fulltext"} and not pmids:
                print(f"SKIP  {name:<14} (search_pubmed returned no PMIDs)")
                failures += 1
                continue
            if name == "get_abstracts":
                args = {"pmids": pmids[:3]}
            elif name == "check_pmids":
                args = {"pmids": pmids[:3] + ["99999999999"]}
            elif name == "get_fulltext":
                args = {"pmcid": pmcid} if pmcid else {"pmid": pmids[0]}
            start = time.perf_counter()
            result = await client.call_tool(name, args)
            ms = (time.perf_counter() - start) * 1000
            data = result.structured_content or {}
            if name == "search_pubmed" and not result.is_error:
                pmids = [a["pmid"] for a in data.get("articles", [])]
                pmcid = next((a["pmcid"] for a in data.get("articles", []) if a.get("pmcid")), None)
            value = data.get(key)
            ok = not result.is_error and bool(value) or (name == "get_fulltext" and "available" in data)
            failures += not ok
            size = len(value) if isinstance(value, list) else value
            print(f"{'PASS' if ok else 'FAIL'}  {name:<14} {ms:7.0f} ms  {key}={size}")
            if not ok:
                print(f"      {result.content[0].text[:300] if result.content else data}")
    return 1 if failures else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true")
    sys.exit(asyncio.run(run(parser.parse_args().offline)))
