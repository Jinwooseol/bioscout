"""The citation judge and run-level judging, offline: a scripted LLM gives the verdicts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from bioscout.answer import CitedAnswer, CitedSentence, Verdict
from bioscout.evals.judging import judge_run, load_judgments, summarize_judgments
from bioscout.evals.metrics import ItemScores, Record
from bioscout.evals.runner import write_report
from bioscout.llm import LLMResult
from bioscout.verifier import JUDGE_SHA, JudgeRefusedError, judge_sentence
from bioscout.verifier.judge import JUDGE_SCHEMA, SYSTEM_JUDGE, CitationJudgment, bound_sentence_verdict
from tests.fakes import FakeLLM, FakePubMed, synthetic_abstract

TEXTS = {
    "10000001": "Drug X lowered mortality compared with drug Y in 400 adults (hazard ratio 0.70).",
    "10000002": "Drug X caused more nausea than placebo.",
    "10000003": "",  # indexed in PubMed without an abstract
}
ABSTRACTS = {pmid: synthetic_abstract(pmid, text) for pmid, text in TEXTS.items()}


def verdict(sentence: str, *citations: tuple[str, str, str], missing: str = "") -> dict[str, Any]:
    """A JUDGE_SCHEMA reply: (pmid, verdict, evidence) per citation."""
    return {
        "citations": [{"pmid": p, "verdict": v, "evidence": e} for p, v, e in citations],
        "sentence_verdict": sentence,
        "missing": missing,
    }


@dataclass
class ByText(FakeLLM):
    """Replies by a phrase in the prompt, so the order in which sentences are judged does not matter."""

    by_text: dict[str, dict[str, Any] | Exception] = field(default_factory=dict)

    async def json(self, **kwargs: Any) -> LLMResult:
        reply = next(r for phrase, r in self.by_text.items() if phrase in kwargs["user"])
        if isinstance(reply, Exception):
            raise reply
        self.replies.append(reply)
        return await super().json(**kwargs)


async def test_judge_gets_the_question_sentence_and_cited_abstracts_and_quotes_are_checked():
    llm = FakeLLM(
        [
            verdict(
                "supported",
                ("10000001", "supported", "Drug X lowered mortality compared with drug Y"),
                ("10000002", "unsupported", "drug X cured nausea"),  # not in the abstract
            )
        ]
    )
    judgment, usage = await judge_sentence(
        llm, "Is X better than Y?", "X lowered mortality versus Y.", ["10000001", " 10000002 "], ABSTRACTS
    )

    call = llm.calls[0]
    assert call["system"] == SYSTEM_JUDGE and call["schema"] == JUDGE_SCHEMA
    assert "Question: Is X better than Y?" in call["user"] and "Sentence: X lowered mortality" in call["user"]
    assert "PMID 10000001" in call["user"] and "hazard ratio 0.70" in call["user"]
    assert judgment.verdict is Verdict.SUPPORTED
    assert [(c.pmid, c.verdict, c.quote_found) for c in judgment.citations] == [
        ("10000001", Verdict.SUPPORTED, True),
        ("10000002", Verdict.UNSUPPORTED, False),
    ]
    assert usage.llm_calls == 1


async def test_citations_without_an_abstract_are_unsupported_without_asking_the_llm():
    llm = FakeLLM([])
    judgment, usage = await judge_sentence(llm, "Q?", "X works.", ["99999999", "10000003"], ABSTRACTS)
    assert llm.calls == [] and usage.llm_calls == 0
    assert judgment.verdict is Verdict.UNSUPPORTED
    assert [(c.pmid, c.verdict, c.reason) for c in judgment.citations] == [
        ("99999999", Verdict.UNSUPPORTED, "pmid_not_found"),
        ("10000003", Verdict.UNSUPPORTED, "no_abstract"),
    ]


async def test_only_usable_abstracts_are_sent_and_odd_replies_are_contained():
    # The judge skips 10000002 and rules on a PMID it was never given.
    llm = FakeLLM(
        [verdict("supported", ("10000001", "partial", "400 adults"), ("55555555", "supported", ""))]
    )
    judgment, _ = await judge_sentence(
        llm, "Q?", "X helps children.", ["PMID: 10000001", "10000002", "99999999", "10000001"], ABSTRACTS
    )
    assert "PMID 99999999" not in llm.calls[0]["user"]
    assert [(c.pmid, c.verdict, c.reason) for c in judgment.citations] == [
        ("10000001", Verdict.PARTIAL, None),
        ("10000002", Verdict.UNSUPPORTED, "not_judged"),
        ("99999999", Verdict.UNSUPPORTED, "pmid_not_found"),
    ]
    assert judgment.verdict is Verdict.PARTIAL  # not better backed than its best citation


@pytest.mark.parametrize(
    ("claimed", "citations", "expected"),
    [
        ("supported", ["partial", "unsupported"], "partial"),
        ("supported", ["unsupported"], "unsupported"),
        ("unsupported", ["supported"], "partial"),
        ("partial", ["supported", "supported"], "partial"),
        ("supported", ["supported", "unsupported"], "supported"),
    ],
)
def test_sentence_verdict_is_bounded_by_its_citations(claimed, citations, expected):
    judged = [CitationJudgment(pmid=str(i), verdict=Verdict(v)) for i, v in enumerate(citations)]
    assert bound_sentence_verdict(Verdict(claimed), judged) is Verdict(expected)


async def test_a_refused_judge_call_raises():
    with pytest.raises(JudgeRefusedError):
        await judge_sentence(FakeLLM([None]), "Q?", "X works.", ["10000001"], ABSTRACTS)


def _record(item_id: str, variant: str, *sentences: tuple[str, list[str]]) -> Record:
    answer = CitedAnswer(
        question="Is X better than Y?",
        decision="yes",
        sentences=[CitedSentence(text=t, citations=c) for t, c in sentences],
    )
    scores = ItemScores(sentences=len(sentences), cited_sentences=0, cited=0, fabricated=0)
    return Record(item_id=item_id, dataset="pubmedqa", variant=variant, answer=answer, scores=scores)


async def test_judging_a_run_writes_verdicts_resumes_and_adds_metrics_to_the_report(tmp_path: Path):
    records = [
        _record(
            "pqa-1",
            "C",
            ("X lowered mortality versus Y.", ["10000001"]),
            ("X lowered mortality and caused nausea.", ["10000001", "99999999"]),
            ("More trials are needed.", []),
        ),
        _record("pqa-2", "C", ("X works in children.", ["10000002"])),
        _record("pqa-1", "A", ("X is widely used.", [])),
    ]
    run_dir = tmp_path / "runs" / "t1"
    run_dir.mkdir(parents=True)
    (run_dir / "records.jsonl").write_text(
        "".join(r.model_dump_json() + "\n" for r in records), encoding="utf-8"
    )
    meta = {"run_id": "t1", "model": "m", "effort": "e"}
    (run_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    quote = "Drug X lowered mortality compared with drug Y"
    llm = ByText(
        [],
        by_text={
            "Sentence: X lowered mortality versus Y.": verdict("supported", ("10000001", "supported", quote)),
            "Sentence: X lowered mortality and caused nausea.": verdict(
                "partial", ("10000001", "supported", quote), missing="nausea"
            ),
            "Sentence: X works in children.": RuntimeError("upstream exploded"),
        },
    )
    pubmed = FakePubMed(texts=TEXTS)
    judged = await judge_run(records, llm, pubmed, run_dir, concurrency=1, progress=lambda _: None)

    assert {key: j.judgment.verdict for key, j in judged.items()} == {
        ("pqa-1", "C", 0): Verdict.SUPPORTED,
        ("pqa-1", "C", 1): Verdict.PARTIAL,
    }
    lines = [
        json.loads(line) for line in (run_dir / "judgments.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(lines) == 3 and all(line["judge_sha"] == JUDGE_SHA for line in lines)
    assert [line["error"] for line in lines if line["error"]] == ["RuntimeError: upstream exploded"]

    # A second call judges only the sentence that failed.
    llm.by_text["Sentence: X works in children."] = verdict("unsupported", ("10000002", "unsupported", ""))
    judged = await judge_run(records, llm, pubmed, run_dir, concurrency=1, progress=lambda _: None)
    assert len(llm.calls) == 3 and len(judged) == 3
    assert load_judgments(run_dir).keys() == judged.keys()

    metrics = summarize_judgments(records, judged)
    c = metrics[("C", "pubmedqa")]
    # Citations: supported, supported, unsupported (99999999 does not exist), unsupported.
    assert c["citations_judged"] == 4 and c["citation_precision"] == pytest.approx(0.5)
    assert c["citation_coverage"] == pytest.approx(1 / 4)  # one of four sentences is fully backed
    assert c["evidence_quote_found_rate"] == 1.0 and c["unjudged_sentences"] == 0
    a = metrics[("A", "pubmedqa")]
    assert a["citation_precision"] is None and a["citation_coverage"] == 0.0

    report = write_report(run_dir, tmp_path / "reports")
    text = report.with_suffix(".md").read_text(encoding="utf-8")
    assert "Citation precision" in text and f"prompt `{JUDGE_SHA}`" in text
    rows = json.loads(report.read_text(encoding="utf-8"))["rows"]
    assert next(r for r in rows if r["variant"] == "C")["citation_precision"] == pytest.approx(0.5)


def test_a_variant_that_was_not_judged_has_no_judge_metrics():
    records = [_record("pqa-1", "C", ("X lowered mortality versus Y.", ["10000001"]))]
    assert summarize_judgments(records, {}) == {("C", "pubmedqa"): {}, ("C", "all"): {}}
