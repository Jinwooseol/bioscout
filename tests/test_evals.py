"""Eval harness, offline: a scripted LLM and an in-memory PubMed with synthetic PMIDs (10000001...)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from bioscout.answer import CitedAnswer, CitedSentence, SearchLog, Usage
from bioscout.cache import ResponseCache
from bioscout.config import CacheMode, Settings
from bioscout.evals.datasets import EvalItem, KeyPoint, load_dataset
from bioscout.evals.metrics import ItemScores, Record, render_markdown, score_item, summarize
from bioscout.evals.runner import load_records, run_eval, write_report
from bioscout.evals.variants import VARIANTS, Context
from bioscout.http import CacheMissError
from bioscout.llm import ClaudeLLM, LLMError, LLMResult
from bioscout.models import AbstractBatch, ArticleAbstract, ArticleSummary, PmidCheck, SearchResult

REAL = {"10000001", "10000002", "10000003"}  # the only PMIDs the fake PubMed knows


@dataclass
class FakeLLM:
    replies: list[dict[str, Any] | None]  # None = refusal
    model: str = "claude-opus-5-5"
    effort: str = "medium"
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def json(self, *, system: str, user: str, schema: dict, max_tokens: int = 16000) -> LLMResult:
        self.calls.append({"system": system, "user": user})
        data = self.replies.pop(0)
        usage = Usage(usd=0.01, llm_calls=1, input_tokens=100, output_tokens=50, model=self.model)
        return LLMResult(data=data, usage=usage, stop_reason="refusal" if data is None else "end_turn")


class FakePubMed:
    def __init__(self, hits: list[str] | None = None) -> None:
        self.hits = hits if hits is not None else ["10000002", "10000001"]
        self.queries: list[str] = []

    async def search(self, query: str, max_results: int = 10) -> SearchResult:
        self.queries.append(query)
        arts = [ArticleSummary(pmid=p, title=f"Synthetic {p}") for p in self.hits[:max_results]]
        return SearchResult(query=query, total_found=len(self.hits), articles=arts)

    async def abstracts(self, pmids: list[str]) -> AbstractBatch:
        return AbstractBatch(articles=[ArticleAbstract(pmid=p, title=f"Synthetic {p}") for p in pmids])

    async def check_pmids(self, pmids: list[str]) -> PmidCheck:
        return PmidCheck(
            existing=[p for p in pmids if p in REAL], missing=[p for p in pmids if p not in REAL]
        )


def reply(decision: str, *sentences: tuple[str, list[str]]) -> dict[str, Any]:
    return {"decision": decision, "sentences": [{"text": t, "citations": c} for t, c in sentences]}


PQA = EvalItem(
    id="pqa-10000001", dataset="pubmedqa", question="Is X better than Y?", answer="yes", pmid="10000001"
)
DOM = EvalItem(
    id="dom-msi-01",
    dataset="domain",
    question="How is MSI detected?",
    must_cite=["10000002", "10000003"],
    key_points=[KeyPoint(point="PCR panel", pmids=["10000002"])],
)


async def test_llm_only_asks_without_tools_and_keeps_citations():
    llm = FakeLLM([reply("yes", ("X beat Y.", [" 10000001 "]), ("Made up.", ["99999999"]))])
    answer = await VARIANTS["A"].run(PQA, Context(llm, FakePubMed()))

    assert answer.decision == "yes"
    assert answer.cited_pmids == ["10000001", "99999999"]
    assert answer.searches == [] and answer.usage.tool_calls == 0
    assert llm.calls[0]["user"] == "Question: Is X better than Y?"


async def test_naive_rag_searches_the_question_verbatim_and_passes_abstracts():
    pubmed = FakePubMed()
    llm = FakeLLM([reply("yes", ("X beat Y.", ["10000001"]))])
    answer = await VARIANTS["B"].run(PQA, Context(llm, pubmed))

    assert pubmed.queries == ["Is X better than Y?"]
    assert answer.searches[0].pmids == ["10000002", "10000001"]
    assert "PMID 10000002 | Synthetic 10000002" in llm.calls[0]["user"]
    assert answer.usage.tool_calls == 2


async def test_naive_rag_with_no_hits_says_so():
    llm = FakeLLM([reply("maybe", ("The search found nothing.", []))])
    answer = await VARIANTS["B"].run(PQA, Context(llm, FakePubMed(hits=[])))
    assert "returned no articles" in llm.calls[0]["user"]
    assert answer.usage.tool_calls == 1


async def test_refusal_becomes_an_empty_refused_answer():
    answer = await VARIANTS["A"].run(PQA, Context(FakeLLM([None]), FakePubMed()))
    scores = await score_item(PQA, answer, FakePubMed())
    assert answer.refused and answer.sentences == []
    assert scores.correct is False and scores.cited == 0


async def test_scores_pubmedqa_item():
    answer = CitedAnswer(
        question=PQA.question,
        decision="no",
        sentences=[
            CitedSentence(text="a", citations=["10000001", "99999999"]),
            CitedSentence(text="b", citations=[]),
        ],
        searches=[SearchLog(query="q", pmids=["10000002", "10000001"])],
    )
    s = await score_item(PQA, answer, FakePubMed())
    assert (s.sentences, s.cited_sentences, s.cited, s.fabricated) == (2, 1, 2, 1)
    assert s.fabricated_pmids == ["99999999"]
    assert s.correct is False and s.retrieved_gold is True and s.gold_cited is True
    assert s.must_cite_recall is None


async def test_scores_domain_item_must_cite_recall():
    answer = CitedAnswer(question=DOM.question, sentences=[CitedSentence(text="a", citations=["10000002"])])
    s = await score_item(DOM, answer, FakePubMed())
    assert s.must_cite_recall == 0.5
    assert s.correct is None and s.retrieved_gold is None


def test_summarize_rates_are_micro_averaged():
    def rec(variant: str, cited: list[str], decision: str) -> Record:
        answer = CitedAnswer(
            question="q", decision=decision, sentences=[CitedSentence(text="s", citations=cited)]
        )
        answer.usage.usd = 0.02
        missing = [p for p in cited if p not in REAL]
        scores = ItemScores(
            sentences=1,
            cited_sentences=int(bool(cited)),
            cited=len(cited),
            fabricated=len(missing),
            correct=decision == "yes",
        )
        return Record(
            item_id=f"i{len(cited)}", dataset="pubmedqa", variant=variant, answer=answer, scores=scores
        )

    records = [
        rec("A", ["10000001", "99999998", "99999999"], "yes"),
        rec("A", ["10000002"], "no"),
        Record(item_id="i9", dataset="pubmedqa", variant="A", error="boom"),
    ]
    [row] = summarize(records)
    assert row["n"] == 3 and row["errors"] == 1
    assert row["fabricated_pmid_rate"] == pytest.approx(2 / 4)
    assert row["answers_with_fabrication"] == pytest.approx(0.5)
    assert row["accuracy"] == pytest.approx(0.5)
    assert row["usd_per_query"] == pytest.approx(0.02)
    assert "| A | pubmedqa | 3 | 50.0% |" in render_markdown(
        [row], {"run_id": "t", "model": "m", "effort": "e"}
    )


async def test_run_writes_records_and_report(tmp_path: Path):
    class FlakyLLM(FakeLLM):
        async def json(self, **kwargs: Any) -> LLMResult:
            if "How is MSI" in kwargs["user"]:
                raise RuntimeError("upstream exploded")
            return await super().json(**kwargs)

    llm = FlakyLLM([reply("yes", ("X beat Y.", ["10000001"]))])
    run_dir = tmp_path / "runs" / "t1"
    records = await run_eval(
        [PQA, DOM],
        [VARIANTS["A"]],
        Context(llm, FakePubMed()),
        run_dir,
        {"run_id": "t1", "model": "m", "effort": "e"},
        progress=lambda _: None,
    )
    assert len(records) == 2
    assert [r.error for r in records if r.error] == ["RuntimeError: upstream exploded"]

    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["variants"]["A"]["prompt_sha"] == VARIANTS["A"].prompt_sha and "finished" in meta
    assert len(load_records(run_dir)) == 2

    report = write_report(run_dir, tmp_path / "reports")
    rows = json.loads(report.read_text(encoding="utf-8"))["rows"]
    assert {(r["variant"], r["dataset"]) for r in rows} == {("A", "pubmedqa"), ("A", "domain"), ("A", "all")}
    assert "Fabricated PMIDs" in report.with_suffix(".md").read_text(encoding="utf-8")


def test_pubmedqa_dataset_file_is_well_formed():
    items = load_dataset("pubmedqa", Path(__file__).parents[1] / "evals" / "datasets")
    assert len(items) == 100
    assert all(i.answer in {"yes", "no", "maybe"} and i.pmid and i.id == f"pqa-{i.pmid}" for i in items)


def test_duplicate_ids_are_rejected(tmp_path: Path):
    line = json.dumps({"id": "x", "question": "q"})
    (tmp_path / "dup.jsonl").write_text(f"{line}\n{line}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_dataset("dup", tmp_path)


# ── ClaudeLLM against a fake Anthropic client ──


def api_response(
    data: dict | None = None, stop: str = "end_turn", model: str = "claude-opus-5-5", iterations=None
) -> SimpleNamespace:
    content = [] if data is None else [SimpleNamespace(type="text", text=json.dumps(data))]
    return SimpleNamespace(
        stop_reason=stop,
        stop_details=SimpleNamespace(category="bio") if stop == "refusal" else None,
        model=model,
        content=content,
        usage=SimpleNamespace(input_tokens=1000, output_tokens=500, iterations=iterations),
    )


class FakeAnthropic:
    def __init__(self, *responses: SimpleNamespace) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create))

    async def create(self, **request: Any) -> SimpleNamespace:
        self.requests.append(request)
        return self.responses.pop(0)


def claude(tmp_path: Path, client: FakeAnthropic, **overrides: Any) -> ClaudeLLM:
    settings = Settings(cache_dir=tmp_path, **overrides)
    return ClaudeLLM(settings, client=client, cache=ResponseCache(tmp_path / "llm.sqlite"))  # type: ignore[arg-type]


async def test_claude_llm_prices_requests_and_replays_from_cache(tmp_path: Path):
    client = FakeAnthropic(api_response({"ok": True}))
    llm = claude(tmp_path, client)

    first = await llm.json(system="s", user="u", schema={"type": "object"})
    second = await llm.json(system="s", user="u", schema={"type": "object"})

    assert len(client.requests) == 1  # the second call came from the cache
    request = client.requests[0]
    assert request["fallbacks"] == "default" and request["output_config"]["effort"] == "medium"
    assert request["output_config"]["format"] == {"type": "json_schema", "schema": {"type": "object"}}
    assert first.data == second.data == {"ok": True}
    assert first.usage.usd == pytest.approx((1000 * 4 + 500 * 20) / 1e6)
    assert (first.usage.cached_llm_calls, second.usage.cached_llm_calls) == (0, 1)


async def test_claude_llm_readonly_miss_raises(tmp_path: Path):
    llm = claude(tmp_path, FakeAnthropic(), cache_mode=CacheMode.READONLY)
    with pytest.raises(CacheMissError):
        await llm.json(system="s", user="u", schema={})


async def test_claude_llm_bills_each_fallback_attempt_at_its_own_rate(tmp_path: Path):
    iterations = [
        SimpleNamespace(type="message", model="claude-opus-5-5", input_tokens=1000, output_tokens=0),
        SimpleNamespace(
            type="fallback_message", model="claude-opus-4-8", input_tokens=1000, output_tokens=500
        ),
    ]
    client = FakeAnthropic(api_response({"ok": 1}, model="claude-opus-4-8", iterations=iterations))
    result = await claude(tmp_path, client).json(system="s", user="u", schema={})
    assert result.usage.fell_back and result.usage.model == "claude-opus-4-8"
    assert result.usage.usd == pytest.approx((1000 * 4 + 1000 * 5 + 500 * 25) / 1e6)


async def test_claude_llm_refusal_and_truncation(tmp_path: Path):
    llm = claude(
        tmp_path, FakeAnthropic(api_response(stop="refusal"), api_response({"a": 1}, stop="max_tokens"))
    )
    refused = await llm.json(system="s", user="refuse me", schema={})
    assert refused.refused and refused.data is None and refused.refusal_category == "bio"
    with pytest.raises(LLMError):
        await llm.json(system="s", user="truncate me", schema={})


async def test_haiku_requests_carry_no_fallback(tmp_path: Path):
    client = FakeAnthropic(api_response({"ok": 1}, model="claude-haiku-5-5"))
    await claude(tmp_path, client, llm_model="claude-haiku-5-5").json(system="s", user="u", schema={})
    assert "fallbacks" not in client.requests[0] and "betas" not in client.requests[0]
