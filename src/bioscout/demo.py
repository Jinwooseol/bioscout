"""What the demo page shows for one question: the answer, with the verdict of the citation check on
every sentence, and the papers it cites.

Everything here is plain data and has no UI code, so the page (app/streamlit_app.py) only lays it out
and this module is tested without Streamlit.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bioscout.agent import ResearchAgent
from bioscout.answer import CitedAnswer, Step, Verdict
from bioscout.config import CacheMode, Settings
from bioscout.llm import LLM, ClaudeLLM
from bioscout.mcp_server import Sources
from bioscout.models import ArticleAbstract
from bioscout.sources.pubmed import Literature
from bioscout.verifier import VerifyConfig

# The questions offered on the page. scripts/build_examples.py records their answers so the page can
# show them without calling Claude.
EXAMPLE_QUESTIONS = [
    "Does MSI-high/mismatch repair-deficient status predict response to PD-1 blockade, and what did "
    "randomized evidence show for first-line treatment of metastatic MSI-high colorectal cancer?",
    "How do mismatch repair protein immunohistochemistry, PCR-based microsatellite analysis, and "
    "next-generation sequencing compare as methods for determining MSI/dMMR status?",
    "What is microsatellite instability, how does loss of DNA mismatch repair produce it, and how does the "
    "origin of sporadic MSI colorectal cancer differ from Lynch syndrome-associated MSI?",
]

VERDICT_LABELS: dict[Verdict | None, str] = {
    Verdict.SUPPORTED: "Supported",
    Verdict.PARTIAL: "Partly supported",
    Verdict.UNSUPPORTED: "Not supported",
    None: "Not checked",
}
DROP_REASONS = {
    "not_backed": "Nothing in the abstracts backs it.",
    "unsupported": "Still not backed after one rewrite.",
}
STEP_LABELS = {
    "plan": "Planned the PubMed searches",
    "search": "Searched PubMed and fetched abstracts",
    "screen": "Read the abstracts and picked the relevant ones",
    "answer": "Wrote the answer",
    "verify": "Checked each sentence against the abstracts it cites",
    "revise": "Rewrote sentences that were not fully backed",
}


@dataclass
class DemoResult:
    answer: CitedAnswer
    # The cited papers the agent fetched, by PMID. Their titles come with the abstracts, so the page
    # needs no further PubMed call and a replayed answer shows the same cards.
    articles: dict[str, ArticleAbstract] = field(default_factory=dict)
    recorded: bool = False  # loaded from the recorded examples, not produced just now


@dataclass
class Source:
    """A cited paper and the sentences (numbered from 1) that cite it."""

    pmid: str
    url: str
    sentences: list[int]
    article: ArticleAbstract | None = None  # None: the agent never fetched a paper with this PMID


def pubmed_url(pmid: str) -> str:
    return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"


async def answer_question(
    llm: LLM,
    pubmed: Literature,
    question: str,
    verify: bool = True,
    on_step: Callable[[Step], None] | None = None,
) -> DemoResult:
    """Run the agent (with the verify step unless `verify` is off) and keep the papers it cites."""
    agent = ResearchAgent(llm, pubmed, verify=VerifyConfig() if verify else None)
    answer, abstracts = await agent.run(question, on_step=on_step)
    articles = {pmid: abstracts[pmid] for pmid in answer.cited_pmids if pmid in abstracts}
    return DemoResult(answer=answer, articles=articles)


async def ask(
    question: str, verify: bool = True, on_step: Callable[[Step], None] | None = None
) -> DemoResult:
    """`answer_question` against the real APIs, with settings from the environment and .env."""
    settings = Settings.from_env()
    sources = Sources(settings)
    try:
        return await answer_question(ClaudeLLM(settings), sources.pubmed, question, verify, on_step)
    finally:
        await sources.http.aclose()


REPO_URL = "https://github.com/Jinwooseol/bioscout"


def can_answer() -> bool:
    """Is there a Claude API key, or a cache to replay answers from without one?"""
    settings = Settings.from_env()
    has_key = any(os.environ.get(k) for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"))
    return has_key or settings.cache_mode is CacheMode.READONLY


def sources_of(result: DemoResult) -> list[Source]:
    """The cited papers in the order the answer first cites them."""
    cited_by: dict[str, list[int]] = {}
    for number, sentence in enumerate(result.answer.sentences, start=1):
        for pmid in sentence.citations:
            cited_by.setdefault(pmid, [])
            if number not in cited_by[pmid]:
                cited_by[pmid].append(number)
    return [
        Source(pmid=pmid, url=pubmed_url(pmid), sentences=numbers, article=result.articles.get(pmid))
        for pmid, numbers in cited_by.items()
    ]


def step_rows(answer: CitedAnswer) -> list[dict[str, Any]]:
    """One row per step for the page's table; a step that ran twice (second search, re-check) is one row."""
    rows: dict[str, dict[str, Any]] = {}
    for step in answer.steps:
        row = rows.setdefault(
            step.name,
            {
                "Step": STEP_LABELS.get(step.name, step.name),
                "Runs": 0,
                "LLM calls": 0,
                "Cost (USD)": 0.0,
                "Seconds": 0.0,
            },
        )
        row["Runs"] += 1
        row["LLM calls"] += step.usage.llm_calls
        row["Cost (USD)"] = round(row["Cost (USD)"] + (step.usage.usd or 0.0), 4)
        row["Seconds"] = round(row["Seconds"] + step.usage.latency_ms / 1000, 1)
    return list(rows.values())


def record(result: DemoResult) -> dict[str, Any]:
    """A result as JSON that may be committed: the answer and the bibliographic line of each cited paper.
    The abstracts, and the passages the judge quoted from them, are left out: article text is never
    committed."""
    sentences = [s.model_copy(update={"evidence": None}) for s in result.answer.sentences]
    answer = result.answer.model_copy(update={"sentences": sentences})
    articles = {
        pmid: {"pmid": a.pmid, "title": a.title, "journal": a.journal, "year": a.year}
        for pmid, a in result.articles.items()
    }
    return {"answer": answer.model_dump(mode="json"), "articles": articles}


def load_recorded(path: Path) -> dict[str, DemoResult]:
    """Recorded answers by question. A missing file means there are none."""
    if not path.exists():
        return {}
    recorded: dict[str, DemoResult] = {}
    for item in json.loads(path.read_text(encoding="utf-8")):
        answer = CitedAnswer.model_validate(item["answer"])
        articles = {pmid: ArticleAbstract(**fields) for pmid, fields in item["articles"].items()}
        recorded[answer.question] = DemoResult(answer=answer, articles=articles, recorded=True)
    return recorded


def was_replayed(answer: CitedAnswer) -> bool:
    """Did every LLM call of this answer come from recorded responses? Then its timings mean nothing."""
    calls = sum(step.usage.llm_calls for step in answer.steps)
    return calls > 0 and calls == sum(step.usage.cached_llm_calls for step in answer.steps)


def totals(answer: CitedAnswer) -> tuple[float, float]:
    """Cost in USD and time in seconds of the whole answer, added up from its steps."""
    usd = sum(step.usage.usd or 0.0 for step in answer.steps)
    seconds = sum(step.usage.latency_ms for step in answer.steps) / 1000
    return usd, seconds
