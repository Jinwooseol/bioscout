"""Judge the citations of a finished run and turn the verdicts into citation precision and coverage.

Judgments go to evals/runs/<run_id>/judgments.jsonl, next to the records and never committed, because
they quote abstracts. Records are not changed, so a new judge prompt can re-judge a run without paying
for the answers again.
"""

from __future__ import annotations

import asyncio
import statistics
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from bioscout.answer import Usage, Verdict
from bioscout.evals.metrics import Record
from bioscout.llm import LLM
from bioscout.models import ArticleAbstract
from bioscout.sources.pubmed import Literature, normalize_pmids
from bioscout.verifier import JUDGE_SHA, SentenceJudgment, judge_sentence

Key = tuple[str, str, int]  # item id, variant, sentence index


class JudgedSentence(BaseModel):
    """One line of judgments.jsonl: the judge's verdict on one cited sentence of one record."""

    item_id: str
    variant: str
    sentence: int = Field(description="Index of the sentence in the answer")
    judge_sha: str
    judgment: SentenceJudgment | None = None
    usage: Usage = Field(default_factory=Usage)
    error: str | None = None


def _append(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(line + "\n")


def load_judgments(run_dir: Path, judge_sha: str = JUDGE_SHA) -> dict[Key, JudgedSentence]:
    """Verdicts made with the given judge prompt. A later line for a sentence replaces an earlier one."""
    path = run_dir / "judgments.jsonl"
    if not path.exists():
        return {}
    judged: dict[Key, JudgedSentence] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entry = JudgedSentence.model_validate_json(line)
            if entry.judge_sha == judge_sha:
                judged[(entry.item_id, entry.variant, entry.sentence)] = entry
    return judged


async def judge_run(
    records: list[Record],
    llm: LLM,
    pubmed: Literature,
    run_dir: Path,
    concurrency: int = 4,
    progress: Callable[[str], None] = print,
) -> dict[Key, JudgedSentence]:
    """Judge every cited sentence that has no verdict yet, appending each as it finishes. A sentence
    that fails is recorded with its error and tried again on the next call."""
    judged = {key: entry for key, entry in load_judgments(run_dir).items() if entry.error is None}
    todo = {
        (r.item_id, r.variant): [
            i
            for i, s in enumerate(r.answer.sentences)
            if s.citations and (r.item_id, r.variant, i) not in judged
        ]
        for r in records
        if r.answer is not None
    }
    total = sum(len(indexes) for indexes in todo.values())
    semaphore = asyncio.Semaphore(concurrency)
    # Appends from two worker threads can land at the same offset on Windows, losing a line.
    write_lock = asyncio.Lock()
    done = 0

    async def one_sentence(
        record: Record, index: int, abstracts: dict[str, ArticleAbstract] | Exception
    ) -> None:
        nonlocal done
        assert record.answer is not None
        sentence = record.answer.sentences[index]
        entry = JudgedSentence(
            item_id=record.item_id, variant=record.variant, sentence=index, judge_sha=JUDGE_SHA
        )
        try:
            if isinstance(abstracts, Exception):
                raise abstracts
            async with semaphore:
                entry.judgment, entry.usage = await judge_sentence(
                    llm, record.answer.question, sentence.text, sentence.citations, abstracts
                )
            judged[(entry.item_id, entry.variant, entry.sentence)] = entry
        except Exception as exc:  # one bad sentence is recorded, it must not sink the whole run
            entry.error = f"{type(exc).__name__}: {exc}"
        async with write_lock:
            await asyncio.to_thread(_append, run_dir / "judgments.jsonl", entry.model_dump_json())
        done += 1
        progress(f"[{done}/{total}] {_progress_line(entry)}")

    async def one_record(record: Record) -> None:
        assert record.answer is not None
        indexes = todo[(record.item_id, record.variant)]
        if not indexes:
            return
        abstracts: dict[str, ArticleAbstract] | Exception
        try:
            async with semaphore:
                batch = await pubmed.abstracts(normalize_pmids(record.answer.cited_pmids))
            abstracts = {a.pmid: a for a in batch.articles}
        except Exception as exc:
            abstracts = exc
        await asyncio.gather(*(one_sentence(record, i, abstracts) for i in indexes))

    await asyncio.gather(*(one_record(r) for r in records if r.answer is not None))
    return judged


def _progress_line(entry: JudgedSentence) -> str:
    head = f"{entry.variant} {entry.item_id} s{entry.sentence}"
    if entry.judgment is None:
        return f"{head} ERROR {(entry.error or '')[:120]}"
    verdicts = " ".join(c.verdict.value for c in entry.judgment.citations)
    return f"{head} {entry.judgment.verdict.value} [{verdicts}]"


def _share(values: list[bool]) -> float | None:
    return statistics.fmean(values) if values else None


def _judge_metrics(records: list[Record], judged: dict[Key, JudgedSentence]) -> dict[str, Any]:
    sentences = supported = unjudged = 0
    verdicts: list[Verdict] = []
    quotes: list[bool] = []
    usd = 0.0
    for r in records:
        assert r.answer is not None
        for i, s in enumerate(r.answer.sentences):
            if not s.citations:
                sentences += 1  # nothing to check it against, so it counts as not backed
                continue
            entry = judged.get((r.item_id, r.variant, i))
            if entry is None or entry.judgment is None:
                unjudged += 1
                continue
            sentences += 1
            supported += entry.judgment.verdict is Verdict.SUPPORTED
            verdicts += [c.verdict for c in entry.judgment.citations]
            quotes += [c.quote_found for c in entry.judgment.citations if c.quote_found is not None]
            usd += entry.usage.usd or 0.0
    if unjudged and not verdicts:
        return {}  # this group was not judged; leave its judge columns empty rather than zero
    return {
        "citations_judged": len(verdicts),
        "citation_precision": _share([v is Verdict.SUPPORTED for v in verdicts]),
        "partial_citation_rate": _share([v is Verdict.PARTIAL for v in verdicts]),
        "citation_coverage": supported / sentences if sentences else None,
        "evidence_quote_found_rate": _share(quotes),
        "unjudged_sentences": unjudged,
        "judge_usd": usd,
    }


def summarize_judgments(
    records: list[Record], judged: dict[Key, JudgedSentence]
) -> dict[tuple[str, str], dict[str, Any]]:
    """Judge metrics per (variant, dataset) and (variant, "all"), keyed like the rows of metrics.summarize."""
    groups: dict[tuple[str, str], list[Record]] = defaultdict(list)
    for r in records:
        if r.answer is not None:
            groups[(r.variant, r.dataset)].append(r)
            groups[(r.variant, "all")].append(r)
    return {key: _judge_metrics(group, judged) for key, group in groups.items()}
