"""Variant D's verify step, offline: judge, prune citations, revise once, drop what nothing backs."""

from __future__ import annotations

from typing import Any

from bioscout.agent import ResearchAgent
from bioscout.agent.prompts import PLAN_SCHEMA, SCREEN_SCHEMA
from bioscout.answer import Verdict
from bioscout.evals.datasets import EvalItem
from bioscout.evals.metrics import Record as EvalRecord
from bioscout.evals.metrics import score_item, summarize_group
from bioscout.evals.variants import VARIANTS, Context
from bioscout.prompts import ANSWER_SCHEMA
from bioscout.verifier import VerifyConfig
from bioscout.verifier.judge import JUDGE_SCHEMA
from bioscout.verifier.revise import REVISE_SCHEMA, SYSTEM_REVISE
from tests.fakes import FakePubMed, RoutedLLM, reply

TEXTS = {
    "10000001": "Drug X lowered mortality compared with drug Y in 400 adults (hazard ratio 0.70).",
    "10000002": "Drug X caused more nausea than placebo.",
}
QUOTE = "Drug X lowered mortality compared with drug Y"


def verdict(sentence: str, *citations: tuple[str, str, str], missing: str = "") -> dict[str, Any]:
    return {
        "citations": [{"pmid": p, "verdict": v, "evidence": e} for p, v, e in citations],
        "sentence_verdict": sentence,
        "missing": missing,
    }


def revised(*sentences: tuple[int, str, list[str]]) -> dict[str, Any]:
    return {"sentences": [{"index": i, "text": t, "citations": c} for i, t, c in sentences]}


def agent(answer: dict[str, Any], judgments: list, revisions: list | None = None):
    """A verifying agent whose search finds both abstracts and whose screener keeps both."""
    llm = RoutedLLM(
        [],
        routes=[
            (PLAN_SCHEMA, [{"searches": [{"purpose": "all", "query": "q1"}]}]),
            (
                SCREEN_SCHEMA,
                [
                    {
                        "relevant_pmids": ["10000001", "10000002"],
                        "enough": True,
                        "missing": "",
                        "next_queries": [],
                    }
                ],
            ),
            (ANSWER_SCHEMA, [answer]),
            (JUDGE_SCHEMA, judgments),
            (REVISE_SCHEMA, revisions or []),
        ],
    )
    return ResearchAgent(llm, FakePubMed(texts=TEXTS), verify=VerifyConfig()), llm


def calls(llm: RoutedLLM, schema: dict) -> list[dict[str, Any]]:
    return [c for c in llm.calls if c["schema"] is schema]


async def test_backed_sentences_keep_only_the_citations_that_back_them():
    answer = reply(
        "yes",
        ("X lowered mortality versus Y.", ["10000001", "10000002", "99999999"]),
        ("X caused nausea.", ["10000002"]),
    )
    judgments = [
        verdict("supported", ("10000001", "supported", QUOTE), ("10000002", "partial", "Drug X")),
        verdict("supported", ("10000002", "supported", "Drug X caused more nausea than placebo")),
    ]
    subject, llm = agent(answer, judgments)
    result = await subject.answer("Is X better than Y?")

    first, second = result.sentences
    assert (first.verdict, first.citations, first.evidence) == (Verdict.SUPPORTED, ["10000001"], QUOTE)
    assert (second.verdict, second.citations) == (Verdict.SUPPORTED, ["10000002"])
    assert result.dropped == [] and result.decision == "yes"
    assert "PMID 99999999" not in calls(llm, JUDGE_SCHEMA)[0]["user"]  # never fetched, so nothing to judge
    assert not calls(llm, REVISE_SCHEMA)
    assert result.usage.llm_calls == 5 and result.cited_pmids == ["10000001", "10000002"]


async def test_a_partly_backed_sentence_is_rewritten_from_the_same_abstracts_and_judged_again():
    answer = reply("yes", ("X halved mortality in children.", ["10000001"]))
    judgments = [
        verdict("partial", ("10000001", "partial", "400 adults"), missing="halved; children"),
        verdict("supported", ("10000001", "supported", QUOTE)),
    ]
    revisions = [revised((0, "X lowered mortality versus Y in adults.", ["10000001", "55555555"]))]
    subject, llm = agent(answer, judgments, revisions)
    result = await subject.answer("Is X better than Y?")

    revise = calls(llm, REVISE_SCHEMA)[0]
    assert revise["system"] == SYSTEM_REVISE
    assert "[0] X halved mortality in children." in revise["user"]
    assert "Not backed: halved; children" in revise["user"]
    assert "PMID 10000002" in revise["user"]  # every abstract the answer was written from is offered
    assert "Sentence: X lowered mortality versus Y in adults." in calls(llm, JUDGE_SCHEMA)[1]["user"]
    (sentence,) = result.sentences
    assert sentence.text == "X lowered mortality versus Y in adults."
    assert (sentence.verdict, sentence.citations) == (
        Verdict.SUPPORTED,
        ["10000001"],
    )  # 55555555 was not given
    assert result.dropped == [] and result.usage.llm_calls == 6


async def test_after_one_revision_partial_stays_flagged_and_unbacked_sentences_are_dropped():
    answer = reply(
        "maybe",
        ("X halved mortality.", ["10000001"]),
        ("X cures nausea.", ["10000002"]),
        ("X is cheap.", ["10000002"]),
    )
    judgments = [
        verdict("partial", ("10000001", "partial", "hazard ratio 0.70"), missing="halved"),
        verdict("unsupported", ("10000002", "unsupported", ""), missing="cures"),
        verdict("unsupported", ("10000002", "unsupported", ""), missing="cost"),
        # After the revision: still only partly backed, and still not backed.
        verdict("partial", ("10000001", "partial", "hazard ratio 0.70"), missing="by how much"),
        verdict("unsupported", ("10000002", "unsupported", "")),
    ]
    revisions = [
        revised(
            (0, "X roughly halved mortality.", ["10000001"]),
            (1, "", []),
            (2, "X is inexpensive.", ["10000002"]),
        )
    ]
    subject, llm = agent(answer, judgments, revisions)
    result = await subject.answer("Is X better than Y?")

    assert [(s.text, s.verdict) for s in result.sentences] == [
        ("X roughly halved mortality.", Verdict.PARTIAL)
    ]
    assert [(d.text, d.reason) for d in result.dropped] == [
        ("X cures nausea.", "not_backed"),  # the reviser found nothing to keep
        ("X is inexpensive.", "unsupported"),  # rewritten, judged again, still not backed
    ]
    assert len(calls(llm, REVISE_SCHEMA)) == 1 and len(calls(llm, JUDGE_SCHEMA)) == 5
    assert result.decision == "maybe"


async def test_a_sentence_without_citations_is_sent_to_revision_without_a_judge_call():
    answer = reply("yes", ("X lowers mortality.", []))
    judgments = [verdict("supported", ("10000001", "supported", QUOTE))]
    revisions = [revised((0, "X lowered mortality versus Y.", ["10000001"]))]
    subject, llm = agent(answer, judgments, revisions)
    result = await subject.answer("Is X better than Y?")

    assert "Cited: nothing" in calls(llm, REVISE_SCHEMA)[0]["user"]
    assert "Not backed: The sentence cites no abstract." in calls(llm, REVISE_SCHEMA)[0]["user"]
    assert len(calls(llm, JUDGE_SCHEMA)) == 1  # only the rewritten sentence is judged
    assert [(s.text, s.verdict) for s in result.sentences] == [
        ("X lowered mortality versus Y.", Verdict.SUPPORTED)
    ]


async def test_a_refused_check_or_revision_leaves_the_sentence_as_it_was():
    answer = reply("yes", ("X lowers mortality.", ["10000001"]), ("X halved mortality.", ["10000001"]))
    judgments = [None, verdict("partial", ("10000001", "partial", "hazard ratio 0.70"), missing="halved")]
    subject, llm = agent(answer, judgments, revisions=[None])
    result = await subject.answer("Is X better than Y?")

    unchecked, partial = result.sentences
    assert (unchecked.text, unchecked.verdict) == ("X lowers mortality.", None)
    assert (partial.text, partial.verdict) == ("X halved mortality.", Verdict.PARTIAL)
    assert result.dropped == [] and result.usage.llm_calls == 6  # refused calls are still counted


async def test_variant_d_is_scored_with_its_dropped_sentences_and_has_its_own_hash():
    item = EvalItem(id="pqa-10000001", dataset="pubmedqa", question="Q?", answer="yes", pmid="10000001")
    answer = reply("yes", ("X lowered mortality versus Y.", ["10000001"]), ("X is cheap.", ["10000002"]))
    judgments = [
        verdict("supported", ("10000001", "supported", QUOTE)),
        verdict("unsupported", ("10000002", "unsupported", "")),
    ]
    subject, _ = agent(answer, judgments, revisions=[revised((1, "", []))])
    pubmed = subject.pubmed
    result = await VARIANTS["D"].run(item, Context(subject.llm, pubmed))
    scores = await score_item(item, result, pubmed)

    assert (scores.sentences, scores.dropped, scores.cited, scores.gold_cited) == (1, 1, 1, True)
    record = EvalRecord(item_id=item.id, dataset="pubmedqa", variant="D", answer=result, scores=scores)
    assert summarize_group([record])["dropped_per_answer"] == 1.0
    # C's hash must not move when the verify step changes; update it only on purpose.
    assert VARIANTS["C"].prompt_sha == "263c2ad233de"
    assert VARIANTS["D"].prompt_sha not in {v.prompt_sha for k, v in VARIANTS.items() if k != "D"}
