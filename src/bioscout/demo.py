"""What the demo page and the HTTP API show for one question: the answer, with the verdict of the
citation check on every sentence, and the papers it cites.

Everything here is plain data and has no UI or web code, so the page (app/streamlit_app.py) and the API
(api.py) only lay it out, and this module is tested without either.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bioscout import tracing
from bioscout.agent import ResearchAgent
from bioscout.answer import CitedAnswer, Step, Verdict
from bioscout.config import CacheMode, Settings
from bioscout.llm import LLM, ClaudeLLM
from bioscout.mcp_server import Sources
from bioscout.models import ArticleAbstract
from bioscout.quota import Quota, QuotaExceeded
from bioscout.sources.pubmed import Literature
from bioscout.tools import open_literature
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

MAX_QUESTION_CHARS = 400  # the page and the API take no longer question

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
    "rank": "Ranked the abstracts by similarity to the question",
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


# Answers one question: (question, verify, on_step) -> the result.
Answerer = Callable[[str, bool, Callable[[Step], None] | None], Awaitable[DemoResult]]


@asynccontextmanager
async def live(settings: Settings | None = None, tags: Sequence[str] = ("demo",)) -> AsyncIterator[Answerer]:
    """`answer_question` against the real APIs, with settings from the environment and .env unless given.
    Every question asked inside the block goes through one HTTP client, and so shares one PubMed rate
    limit: a server keeps the block open for as long as it runs."""
    settings = settings or Settings.from_env()
    sources = Sources(settings)
    tracing.configure(settings)
    llm = tracing.traced_llm(ClaudeLLM(settings))
    try:
        async with open_literature(sources, settings.through_mcp) as pubmed:
            traced = tracing.traced_literature(pubmed)

            async def answer(
                question: str, verify: bool = True, on_step: Callable[[Step], None] | None = None
            ) -> DemoResult:
                with tracing.context(tags=list(tags)):
                    return await answer_question(llm, traced, question, verify, on_step)

            yield answer
    finally:
        await sources.http.aclose()


async def ask(
    question: str, verify: bool = True, on_step: Callable[[Step], None] | None = None
) -> DemoResult:
    """Answer one question against the real APIs."""
    async with live() as answer:
        return await answer(question, verify, on_step)


REPO_URL = "https://github.com/Jinwooseol/bioscout"


def can_answer(settings: Settings | None = None) -> bool:
    """Is there a Claude API key, or a cache to replay answers from without one?"""
    settings = settings or Settings.from_env()
    has_key = any(os.environ.get(k) for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"))
    return has_key or settings.cache_mode is CacheMode.READONLY


def quota() -> Quota:
    """The allowance of questions that the page and the API both draw on (quota.py)."""
    return Quota.from_settings(Settings.from_env())


def quota_text(exceeded: QuotaExceeded) -> str:
    """What a visitor is told when the allowance is used up."""
    if exceeded.scope == "day":
        return (
            "Today's allowance of live questions, shared by all visitors, is used up. It starts again at "
            f"{exceeded.status.resets_at:%H:%M} UTC. The example questions still show recorded answers."
        )
    return (
        "The allowance of live questions for this deployment is used up. The example questions still "
        f"show recorded answers. To ask your own, run BioScout with your own key: {REPO_URL}"
    )


def failure_text(exc: BaseException) -> str:
    """What a visitor is told when an answer fails: the kind of error only. The text of an exception
    can quote a request, and a request to PubMed carries the API key and the contact address."""
    return f"The answer could not be produced ({type(exc).__name__})."


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


@dataclass
class StepTotal:
    """What one kind of step added up to; a step that ran twice (second search, re-check) is one total."""

    name: str
    label: str
    runs: int = 0
    llm_calls: int = 0
    usd: float = 0.0
    seconds: float = 0.0


def step_totals(answer: CitedAnswer) -> list[StepTotal]:
    totals: dict[str, StepTotal] = {}
    for step in answer.steps:
        total = totals.setdefault(step.name, StepTotal(step.name, STEP_LABELS.get(step.name, step.name)))
        total.runs += 1
        total.llm_calls += step.usage.llm_calls
        total.usd = round(total.usd + (step.usage.usd or 0.0), 4)
        total.seconds = round(total.seconds + step.usage.latency_ms / 1000, 1)
    return list(totals.values())


def step_rows(answer: CitedAnswer) -> list[dict[str, Any]]:
    """One row per kind of step for the page's table."""
    return [
        {"Step": t.label, "Runs": t.runs, "LLM calls": t.llm_calls, "Cost (USD)": t.usd, "Seconds": t.seconds}
        for t in step_totals(answer)
    ]


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
