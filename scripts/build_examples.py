"""Record the demo's example answers, so the page can show them without calling Claude.

    uv run python scripts/build_examples.py

Runs each example question through the verifying agent and writes app/examples.json. The file holds the
answers and the bibliographic line of each cited paper; abstracts and the passages quoted from them are
left out. Questions already in the response cache replay for free, anything else is paid for; with
BIOSCOUT_CACHE_MODE=readonly nothing is spent and an uncached question fails instead.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from bioscout import demo

OUT = Path(__file__).parent.parent / "app" / "examples.json"


async def main() -> None:
    records = []
    for question in demo.EXAMPLE_QUESTIONS:
        result = await demo.ask(question)
        usd, _ = demo.totals(result.answer)
        how = "replayed from the cache" if demo.was_replayed(result.answer) else "run now"
        print(f"{len(result.answer.sentences)} sentences, {len(result.articles)} papers, ${usd:.2f}, {how}")
        print(f"  {question[:90]}")
        records.append(demo.record(result))
    OUT.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
