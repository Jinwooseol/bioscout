"""Compare ways of choosing the abstracts an answer is written from, without writing any answer.

The agent is run up to its choice of abstracts and stops. What is counted is whether the paper each
question should cite (the source paper of a PubMedQA question, every must-cite paper of a domain
question) is among the abstracts chosen. With the cache read-only the model calls of a variant that
was run before are replayed, so the comparison costs nothing.
"""

from __future__ import annotations

import asyncio
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from bioscout.agent import ResearchAgent
from bioscout.agent.rank import Ranker
from bioscout.answer import Usage
from bioscout.evals.datasets import EvalItem
from bioscout.evals.metrics import format_cell
from bioscout.llm import LLM
from bioscout.sources.pubmed import Literature

# The variants whose choice of abstracts differs. D chooses as C does.
SELECTORS = ("C", "E")

COLUMNS = [
    ("variant", "Variant", "str"),
    ("dataset", "Dataset", "str"),
    ("n", "n", "str"),
    ("papers", "Papers", "str"),
    ("found", "Among the candidates", "str"),
    ("selected", "Chosen", "str"),
    ("selected_rate", "Chosen %", "pct"),
    ("context_size", "Abstracts chosen / question", "num"),
    ("same_as_first", "Shared with the first variant", "pct"),
    ("llm_calls_per_query", "LLM calls / question", "num"),
    ("usd_per_query", "$ / question, up to the choice", "usd"),
    ("choose_ms_p50", "Choosing, p50", "ms"),
    ("errors", "Errors", "str"),
]


@dataclass
class Selection:
    """What one variant chose for one question."""

    item_id: str
    dataset: str
    variant: str
    candidates: list[str] = field(default_factory=list)  # every abstract fetched
    selected: list[str] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    choose_ms: int | None = None  # the choosing step, when it ran rather than replayed
    error: str | None = None


def build_agent(variant: str, llm: LLM, pubmed: Literature, ranker: Ranker | None) -> ResearchAgent:
    if variant == "E":
        assert ranker is not None
        return ResearchAgent(llm, pubmed, ranker=ranker)
    return ResearchAgent(llm, pubmed)


async def select_one(item: EvalItem, variant: str, agent: ResearchAgent) -> Selection:
    selection = Selection(item_id=item.id, dataset=item.dataset, variant=variant)
    try:
        state = await agent.select(item.question)
    except Exception as exc:  # one failed question is counted, it must not sink the comparison
        selection.error = f"{type(exc).__name__}: {exc}"
        return selection
    selection.candidates = list(state.get("abstracts", {}))
    selection.selected = list(state.get("selected", []))
    selection.usage = Usage.total(state.get("usages", []))
    chose = [
        s for s in state.get("steps", []) if s.name in ("screen", "rank") and not s.usage.cached_llm_calls
    ]
    if chose:
        selection.choose_ms = sum(step.usage.latency_ms for step in chose)
    return selection


async def select_all(
    items: list[EvalItem],
    agents: dict[str, ResearchAgent],
    concurrency: int = 4,
    progress: Callable[[str], None] = print,
) -> list[Selection]:
    semaphore = asyncio.Semaphore(concurrency)
    total = len(items) * len(agents)
    done = 0
    started = time.perf_counter()

    async def one(item: EvalItem, variant: str) -> Selection:
        nonlocal done
        async with semaphore:
            selection = await select_one(item, variant, agents[variant])
        done += 1
        if done % 25 == 0 or done == total:
            progress(f"[{done}/{total}] {time.perf_counter() - started:.0f} s")
        return selection

    return list(await asyncio.gather(*(one(item, variant) for variant in agents for item in items)))


def expected_papers(item: EvalItem) -> list[str]:
    return [item.pmid] if item.pmid else list(item.must_cite)


def summarize_selection(selections: list[Selection], items: dict[str, EvalItem]) -> list[dict[str, Any]]:
    """One row per (variant, dataset), in the order the variants were given."""
    first = selections[0].variant if selections else ""
    reference = {s.item_id: set(s.selected) for s in selections if s.variant == first and not s.error}
    groups: dict[tuple[str, str], list[Selection]] = {}
    for selection in selections:
        groups.setdefault((selection.variant, selection.dataset), []).append(selection)

    rows = []
    for (variant, dataset), group in groups.items():
        ok = [s for s in group if not s.error]
        papers = [(s, pmid) for s in ok for pmid in expected_papers(items[s.item_id])]
        chosen = sum(pmid in s.selected for s, pmid in papers)
        shared = [
            len(set(s.selected) & reference[s.item_id]) / len(reference[s.item_id])
            for s in ok
            if reference.get(s.item_id)
        ]
        timed = [s.choose_ms for s in ok if s.choose_ms is not None]
        usd = [s.usage.usd for s in ok]
        rows.append(
            {
                "variant": variant,
                "dataset": dataset,
                "n": len(group),
                "papers": len(papers),
                "found": sum(pmid in s.candidates for s, pmid in papers),
                "selected": chosen,
                "selected_rate": chosen / len(papers) if papers else None,
                "context_size": statistics.mean(len(s.selected) for s in ok) if ok else None,
                "same_as_first": statistics.mean(shared) if shared and variant != first else None,
                "llm_calls_per_query": statistics.mean(s.usage.llm_calls for s in ok) if ok else None,
                "usd_per_query": None if not ok or None in usd else statistics.mean(usd),
                "choose_ms_p50": statistics.median(timed) if timed else None,
                "errors": len(group) - len(ok),
            }
        )
    return rows


def render_selection(rows: list[dict[str, Any]], meta: dict[str, Any]) -> str:
    head = "| " + " | ".join(label for _, label, _ in COLUMNS) + " |"
    rule = "|" + "|".join(" --- " for _ in COLUMNS) + "|"
    body = [
        "| " + " | ".join(format_cell(row.get(key), kind) for key, _, kind in COLUMNS) + " |" for row in rows
    ]
    lines = [
        f"# Choice of abstracts {meta['run_id']}",
        "",
        f"Model `{meta['model']}`, effort `{meta['effort']}`, ranker `{meta.get('ranker') or 'none'}`, "
        f"cache `{meta['cache_mode']}`, commit `{meta.get('commit') or 'unknown'}`.",
        "The agent is run up to its choice of abstracts and no answer is written. Papers: the source paper "
        "of each PubMedQA question, every must-cite paper of each domain question. A paper is among the "
        "candidates when a search returned it and its abstract was fetched; it is chosen when it is among "
        "the abstracts the answer step would be given. Model calls that were replayed from the cache are "
        "counted at what they cost when first made, and their time is not counted.",
        "",
        head,
        rule,
        *body,
        "",
    ]
    return "\n".join(lines)
