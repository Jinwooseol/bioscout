"""Judge calibration, offline: sampling, the labeling file round trip, Cohen's kappa and scoring."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pytest

from bioscout.answer import CitedAnswer, CitedSentence, Verdict
from bioscout.evals.calibration import (
    Pair,
    PairSet,
    cohen_kappa,
    read_labels,
    render_labeling_file,
    sample_pairs,
    score_labels,
)
from bioscout.evals.cli import cmd_calibrate_score
from bioscout.evals.judging import JudgedSentence, Key
from bioscout.evals.metrics import ItemScores, Record
from bioscout.verifier import JUDGE_SHA, CitationJudgment, SentenceJudgment
from tests.fakes import synthetic_abstract

S, P, U = Verdict.SUPPORTED, Verdict.PARTIAL, Verdict.UNSUPPORTED


def judged_sentence(
    item_id: str, variant: str, *citations: tuple[str, Verdict, str | None]
) -> JudgedSentence:
    """A judged first sentence; each citation is (pmid, verdict, reason for having no LLM verdict)."""
    verdicts = [CitationJudgment(pmid=p, verdict=v, reason=r) for p, v, r in citations]
    judgment = SentenceJudgment(verdict=citations[0][1], citations=verdicts)
    return JudgedSentence(
        item_id=item_id, variant=variant, sentence=0, judge_sha=JUDGE_SHA, judgment=judgment
    )


def as_dict(entries: list[JudgedSentence]) -> dict[Key, JudgedSentence]:
    return {(e.item_id, e.variant, e.sentence): e for e in entries}


def many(variant: str, verdict: Verdict, count: int) -> list[JudgedSentence]:
    return [
        judged_sentence(f"{variant}-{verdict.value}-{i}", variant, (f"{i + 1}", verdict, None))
        for i in range(count)
    ]


JUDGED = as_dict(
    [
        *many("A", S, 10),
        *many("B", S, 10),
        *many("C", S, 10),
        *many("A", P, 2),
        *many("B", P, 10),
        *many("C", P, 10),
        *many("A", U, 3),
        *many("C", U, 1),
        judged_sentence("no-abstract", "A", ("99999999", U, "no_abstract")),
    ]
)


def verdict_of(pair: Pair) -> Verdict:
    entry = JUDGED[(pair.item_id, pair.variant, pair.sentence)]
    return next(c.verdict for c in entry.judgment.citations if c.pmid == pair.pmid)


def test_sample_takes_the_same_number_per_verdict_and_turns_between_variants():
    pairs = sample_pairs(JUDGED, n=12, seed=1)
    assert [p.n for p in pairs] == list(range(1, 13))
    assert Counter(verdict_of(p) for p in pairs) == {S: 4, P: 4, U: 4}
    by_verdict = {v: Counter(p.variant for p in pairs if verdict_of(p) is v) for v in Verdict}
    assert by_verdict[S] == {"A": 2, "B": 1, "C": 1}
    assert by_verdict[P] == {"A": 2, "B": 1, "C": 1}
    assert by_verdict[U] == {"A": 3, "C": 1}  # every unsupported citation there is
    assert all(p.pmid != "99999999" for p in pairs)  # nothing to label without an abstract


def test_a_short_verdict_passes_its_share_on_and_the_sample_is_repeatable():
    pairs = sample_pairs(JUDGED, n=15, seed=1)
    assert Counter(verdict_of(p) for p in pairs) == {S: 6, P: 5, U: 4}
    assert sample_pairs(JUDGED, n=15, seed=1) == pairs
    assert sample_pairs(JUDGED, n=15, seed=2) != pairs
    assert len(sample_pairs(JUDGED, n=500, seed=1)) == 56  # never more than there is


def _answer_record(item_id: str, variant: str, text: str, pmid: str) -> Record:
    answer = CitedAnswer(question="Is X worth using?", sentences=[CitedSentence(text=text, citations=[pmid])])
    scores = ItemScores(sentences=1, cited_sentences=1, cited=1, fabricated=0)
    return Record(item_id=item_id, dataset="pubmedqa", variant=variant, answer=answer, scores=scores)


RECORDS = [
    _answer_record("pqa-1", "A", "X lowers mortality.", "10000001"),
    _answer_record("pqa-2", "B", "X causes nausea.", "10000002"),
    _answer_record("pqa-3", "C", "X works in children.", "10000001"),
    _answer_record("pqa-4", "C", "X is cheap.", "10000002"),
]
ABSTRACTS = {
    "10000001": synthetic_abstract("10000001", "Drug X lowered mortality in adults."),
    "10000002": synthetic_abstract("10000002", "Drug X caused more nausea than placebo."),
}
PAIRS = PairSet(
    run_id="t1",
    judge_sha=JUDGE_SHA,
    seed=1,
    pairs=[
        Pair(n=i, item_id=r.item_id, variant=r.variant, sentence=0, pmid=r.answer.sentences[0].citations[0])
        for i, r in enumerate(RECORDS, start=1)
    ],
)
PAIR_VERDICTS = as_dict(
    [
        judged_sentence("pqa-1", "A", ("10000001", S, None)),
        judged_sentence("pqa-2", "B", ("10000002", S, None)),
        judged_sentence("pqa-3", "C", ("10000001", P, None)),
        judged_sentence("pqa-4", "C", ("10000002", U, None)),
    ]
)


def _fill(text: str, *labels: str) -> str:
    for label in labels:
        text = text.replace("label:\n", f"label: {label}\n" if label else "label: \n", 1)
    return text


def test_labeling_file_shows_the_pair_but_not_the_verdict_or_the_variant():
    text = render_labeling_file(PAIRS, RECORDS, ABSTRACTS)
    header, *sections = text.split("\n---\n")
    assert "calibrate-score evals/runs/t1" in header and len(sections) == 4
    first = sections[0]
    assert "## 01" in first and "Is X worth using?" in first and "X lowers mortality." in first
    assert "PMID 10000001" in first and "Drug X lowered mortality in adults." in first
    for section in sections:
        assert not any(v.value in section for v in Verdict)
        assert not any(r.item_id in section for r in RECORDS)


def test_labels_are_read_back_whatever_the_case_or_line_endings():
    text = _fill(render_labeling_file(PAIRS, RECORDS, ABSTRACTS), "s", "P", "unsupported", "")
    expected = {1: S, 2: P, 3: U, 4: None}
    assert read_labels(text) == expected
    assert read_labels(text.replace("\n", "\r\n")) == expected


def test_an_unknown_label_is_reported_with_its_pair_number():
    text = _fill(render_labeling_file(PAIRS, RECORDS, ABSTRACTS), "s", "maybe")
    with pytest.raises(ValueError, match="#02: 'maybe'"):
        read_labels(text)


def test_cohen_kappa_matches_a_worked_example():
    rated = [("y", "y")] * 20 + [("y", "n")] * 5 + [("n", "y")] * 10 + [("n", "n")] * 15
    assert cohen_kappa(rated) == pytest.approx(0.4)  # observed 0.7, expected by chance 0.5
    assert cohen_kappa([("y", "y"), ("n", "n")]) == 1.0
    assert cohen_kappa([("y", "y")] * 3) == 1.0  # one label only: nothing to disagree on
    assert cohen_kappa([]) is None


def test_scoring_counts_agreement_and_lists_disagreements():
    result, disagreements = score_labels(PAIRS, {1: S, 2: P, 3: P, 4: None}, PAIR_VERDICTS)
    assert (result.pairs, result.labelled) == (4, 3)
    assert result.agreement == pytest.approx(2 / 3) and result.kappa == pytest.approx(0.4)
    assert result.agreement_supported_vs_rest == pytest.approx(2 / 3)
    assert result.confusion["partial"] == {"supported": 1, "partial": 1, "unsupported": 0}
    assert [(d.n, d.human, d.judge, d.variant) for d in disagreements] == [(2, P, S, "B")]


def _run_with_labels(tmp_path: Path, *labels: str) -> argparse.Namespace:
    run_dir = tmp_path / "runs" / "t1"
    run_dir.mkdir(parents=True)
    (run_dir / "meta.json").write_text(json.dumps({"run_id": "t1", "model": "m", "effort": "e"}))
    (run_dir / "records.jsonl").write_text("".join(r.model_dump_json() + "\n" for r in RECORDS))
    (run_dir / "judgments.jsonl").write_text(
        "".join(j.model_dump_json() + "\n" for j in PAIR_VERDICTS.values())
    )
    review = tmp_path / "review"
    review.mkdir()
    text = _fill(render_labeling_file(PAIRS, RECORDS, ABSTRACTS), *labels)
    (review / "calibration.md").write_text(text, encoding="utf-8")
    (review / "calibration.pairs.json").write_text(PAIRS.model_dump_json(), encoding="utf-8")
    return argparse.Namespace(run_dir=run_dir, evals_dir=tmp_path, labels=None)


def test_the_result_is_saved_and_noted_in_the_report_once_every_pair_is_labelled(tmp_path: Path, capsys):
    args = _run_with_labels(tmp_path, "s", "s", "p", "u")
    assert cmd_calibrate_score(args) == 0
    saved = json.loads((tmp_path / "reports" / "t1.calibration.json").read_text(encoding="utf-8"))
    assert saved["kappa"] == 1.0 and saved["labelled"] == 4 and saved["judge_sha"] == JUDGE_SHA
    assert "abstract" not in json.dumps(saved).lower()  # numbers only: this file is committed
    assert "Cohen's kappa 1.00" in (tmp_path / "reports" / "t1.md").read_text(encoding="utf-8")
    assert "meets the 0.70 target" in capsys.readouterr().out


def test_an_unfinished_labeling_file_is_scored_but_not_saved(tmp_path: Path, capsys):
    args = _run_with_labels(tmp_path, "s", "p")
    assert cmd_calibrate_score(args) == 1
    out = capsys.readouterr().out
    assert "labelled 2 of 4 pairs" in out and "#02 you=partial judge=supported" in out
    assert not (tmp_path / "reports").exists()
