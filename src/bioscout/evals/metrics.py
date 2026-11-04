"""Deterministic metrics: everything that needs no LLM judge. Judge metrics (citation precision,
coverage, key-point coverage) are added on Day 4 on top of the same records."""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any

from pydantic import BaseModel, Field

from bioscout.answer import CitedAnswer
from bioscout.evals.datasets import EvalItem
from bioscout.evals.variants import Literature
from bioscout.models import PmidCheck


class ItemScores(BaseModel):
    sentences: int
    cited_sentences: int = Field(description="Sentences with at least one citation")
    cited: int = Field(description="Unique cited PMIDs")
    fabricated: int = Field(description="Cited PMIDs that PubMed does not know")
    fabricated_pmids: list[str] = Field(default_factory=list)
    correct: bool | None = Field(default=None, description="PubMedQA: decision matches the label")
    retrieved_gold: bool | None = Field(default=None, description="PubMedQA: source paper in a top 10")
    gold_cited: bool | None = Field(default=None, description="PubMedQA: source paper cited")
    must_cite_recall: float | None = Field(default=None, description="Domain: share of must-cite cited")


class Record(BaseModel):
    """One line of evals/runs/<run_id>/records.jsonl: one item answered by one variant."""

    item_id: str
    dataset: str
    variant: str
    answer: CitedAnswer | None = None
    scores: ItemScores | None = None
    error: str | None = None


async def score_item(item: EvalItem, answer: CitedAnswer, pubmed: Literature) -> ItemScores:
    cited = answer.cited_pmids
    check = await pubmed.check_pmids(cited) if cited else PmidCheck(existing=[], missing=[])
    existing = set(check.existing)

    scores = ItemScores(
        sentences=len(answer.sentences),
        cited_sentences=sum(1 for s in answer.sentences if s.citations),
        cited=len(check.existing) + len(check.missing),
        fabricated=len(check.missing),
        fabricated_pmids=check.missing,
    )
    if item.answer is not None:
        scores.correct = answer.decision == item.answer
    if item.pmid is not None:
        scores.gold_cited = item.pmid in existing
        if answer.searches:
            scores.retrieved_gold = any(item.pmid in s.pmids[:10] for s in answer.searches)
    if item.must_cite:
        scores.must_cite_recall = len(set(item.must_cite) & existing) / len(set(item.must_cite))
    return scores


def _mean(values: list[float | bool]) -> float | None:
    return statistics.fmean(values) if values else None


def _percentile(values: list[int], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return float(ordered[index])


def summarize_group(records: list[Record]) -> dict[str, Any]:
    ok = [r for r in records if r.scores is not None and r.answer is not None]
    scores = [r.scores for r in ok if r.scores]
    answers = [r.answer for r in ok if r.answer]
    cited = sum(s.cited for s in scores)
    sentences = sum(s.sentences for s in scores)
    usd = [a.usage.usd for a in answers]
    llm_calls = sum(a.usage.llm_calls for a in answers)
    return {
        "n": len(records),
        "errors": len(records) - len(ok),
        "refusals": sum(a.refused for a in answers),
        "fell_back": sum(a.usage.fell_back for a in answers),
        "fabricated_pmid_rate": sum(s.fabricated for s in scores) / cited if cited else None,
        "answers_with_fabrication": _mean([s.fabricated > 0 for s in scores]),
        "cited_per_answer": _mean([s.cited for s in scores]),
        "citation_rate": sum(s.cited_sentences for s in scores) / sentences if sentences else None,
        "accuracy": _mean([s.correct for s in scores if s.correct is not None]),
        "retrieval_recall_at_10": _mean([s.retrieved_gold for s in scores if s.retrieved_gold is not None]),
        "gold_cited_rate": _mean([s.gold_cited for s in scores if s.gold_cited is not None]),
        "must_cite_recall": _mean([s.must_cite_recall for s in scores if s.must_cite_recall is not None]),
        "usd_per_query": None if None in usd else _mean(usd),
        "usd_total": None if None in usd else sum(usd),
        "latency_p50_ms": _percentile([a.usage.latency_ms for a in answers], 0.5),
        "latency_p95_ms": _percentile([a.usage.latency_ms for a in answers], 0.95),
        "tool_calls_per_query": _mean([a.usage.tool_calls for a in answers]),
        "cached_share": sum(a.usage.cached_llm_calls for a in answers) / llm_calls if llm_calls else None,
    }


def summarize(records: list[Record]) -> list[dict[str, Any]]:
    """One row per (variant, dataset), plus an 'all' row per variant when there are several datasets."""
    groups: dict[tuple[str, str], list[Record]] = defaultdict(list)
    for r in records:
        groups[(r.variant, r.dataset)].append(r)
        groups[(r.variant, "all")].append(r)
    datasets = {r.dataset for r in records}
    rows = []
    for (variant, dataset), group in sorted(groups.items()):
        if dataset == "all" and len(datasets) < 2:
            continue
        rows.append({"variant": variant, "dataset": dataset, **summarize_group(group)})
    return rows


def _fmt(value: Any, kind: str) -> str:
    if value is None:
        return "—"
    if kind == "pct":
        return f"{value * 100:.1f}%"
    if kind == "usd":
        return f"${value:.3f}"
    if kind == "ms":
        return f"{value / 1000:.1f}s"
    if kind == "num":
        return f"{value:.2f}"
    return str(value)


COLUMNS = [
    ("variant", "Variant", "str"),
    ("dataset", "Dataset", "str"),
    ("n", "n", "str"),
    ("fabricated_pmid_rate", "Fabricated PMIDs", "pct"),
    ("answers_with_fabrication", "Answers w/ fabrication", "pct"),
    ("citation_rate", "Sentences cited", "pct"),
    ("accuracy", "Accuracy", "pct"),
    ("retrieval_recall_at_10", "Recall@10", "pct"),
    ("gold_cited_rate", "Source cited", "pct"),
    ("must_cite_recall", "Must-cite recall", "pct"),
    ("usd_per_query", "$/query", "usd"),
    ("latency_p50_ms", "p50", "ms"),
    ("errors", "Errors", "str"),
    ("refusals", "Refusals", "str"),
]


def render_markdown(rows: list[dict[str, Any]], meta: dict[str, Any]) -> str:
    head = "| " + " | ".join(label for _, label, _ in COLUMNS) + " |"
    rule = "|" + "|".join(" --- " for _ in COLUMNS) + "|"
    body = ["| " + " | ".join(_fmt(row.get(key), kind) for key, _, kind in COLUMNS) + " |" for row in rows]
    lines = [
        f"# Eval run {meta['run_id']}",
        "",
        f"Model `{meta['model']}`, effort `{meta['effort']}`, commit `{meta.get('commit') or 'unknown'}`.",
        "Deterministic metrics only; judge metrics (citation precision, coverage) arrive on Day 4.",
        "",
        head,
        rule,
        *body,
        "",
    ]
    return "\n".join(lines)
