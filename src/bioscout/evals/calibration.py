"""Calibrating the judge against hand labels.

Sample (sentence, cited abstract) pairs from a judged run, write them to a file to label without seeing
the judge's verdict, then score the judge against the labels (agreement, Cohen's kappa, confusion).

The labeling file quotes abstracts, so it lives in evals/review/ and is never committed. The result is
numbers only and goes to evals/reports/<run_id>.calibration.json.
"""

from __future__ import annotations

import random
import re
from collections import Counter, defaultdict

from pydantic import BaseModel, Field

from bioscout.answer import Verdict
from bioscout.evals.judging import JudgedSentence, Key
from bioscout.evals.metrics import Record
from bioscout.models import ArticleAbstract
from bioscout.prompts import format_abstracts

LABELS = {
    "s": Verdict.SUPPORTED,
    "supported": Verdict.SUPPORTED,
    "p": Verdict.PARTIAL,
    "partial": Verdict.PARTIAL,
    "u": Verdict.UNSUPPORTED,
    "unsupported": Verdict.UNSUPPORTED,
}
KAPPA_TARGET = 0.7


class Pair(BaseModel):
    """One (sentence, cited abstract) pair to label; `n` is its number in the labeling file."""

    n: int
    item_id: str
    variant: str
    sentence: int
    pmid: str


class PairSet(BaseModel):
    """Written next to the labeling file. It says which pair is which, and holds no verdict."""

    run_id: str
    judge_sha: str
    seed: int
    pairs: list[Pair]


class Disagreement(BaseModel):
    n: int
    item_id: str
    variant: str
    pmid: str
    human: Verdict
    judge: Verdict


class Calibration(BaseModel):
    """Numbers only, so it can be committed with the reports."""

    run_id: str
    judge_sha: str
    seed: int
    pairs: int
    labelled: int
    agreement: float | None = None
    kappa: float | None = Field(default=None, description="Cohen's kappa over the three verdicts")
    agreement_supported_vs_rest: float | None = None
    kappa_supported_vs_rest: float | None = None
    confusion: dict[str, dict[str, int]] = Field(
        default_factory=dict, description="Counts by hand label, then by judge verdict"
    )


def _quotas(sizes: dict[Verdict, int], n: int) -> dict[Verdict, int]:
    """Equal shares per verdict; a verdict with too few citations passes the rest of its share on."""
    quota = dict.fromkeys(sizes, 0)
    left = min(n, sum(sizes.values()))
    while left:
        for verdict in sizes:
            if left and quota[verdict] < sizes[verdict]:
                quota[verdict] += 1
                left -= 1
    return quota


def sample_pairs(judged: dict[Key, JudgedSentence], n: int = 60, seed: int = 2026) -> list[Pair]:
    """Pick citations the LLM judged, the same number per judge verdict and, within a verdict, taking
    turns between variants. Most citations are judged supported, so a plain random sample would hold
    almost no unsupported ones and could not show whether the judge catches wrong citations.
    """
    rng = random.Random(seed)
    pools: dict[Verdict, dict[str, list[tuple[str, str, int, str]]]] = {v: defaultdict(list) for v in Verdict}
    for (item_id, variant, index), entry in sorted(judged.items()):
        if entry.judgment is None:
            continue
        for citation in entry.judgment.citations:
            if citation.reason is None:  # without an abstract there is nothing to label
                pools[citation.verdict][variant].append((item_id, variant, index, citation.pmid))

    quota = _quotas({v: sum(len(pool) for pool in pools[v].values()) for v in Verdict}, n)
    picked: list[tuple[str, str, int, str]] = []
    for verdict in Verdict:
        by_variant = pools[verdict]
        for pool in by_variant.values():
            rng.shuffle(pool)
        variants = sorted(by_variant)
        turn = 0
        for _ in range(quota[verdict]):
            while not by_variant[variants[turn % len(variants)]]:
                turn += 1
            picked.append(by_variant[variants[turn % len(variants)]].pop())
            turn += 1
    rng.shuffle(picked)  # so the order of the file says nothing about the verdict
    return [
        Pair(n=i, item_id=item_id, variant=variant, sentence=index, pmid=pmid)
        for i, (item_id, variant, index, pmid) in enumerate(picked, start=1)
    ]


HEADER = """\
# Judge calibration: {count} pairs to label

Run `{run_id}` · judge prompt `{judge_sha}` · seed {seed}

Each pair is one sentence from an answer and the abstract of one paper that the sentence cites. Read the
abstract and decide whether it backs the sentence, then write `s`, `p` or `u` after `label:`. The judge's
verdict and the variant that wrote the answer are left out on purpose.

- `s` (supported): the abstract clearly states at least one factual claim that the sentence makes, with
  the same direction of effect and, where the sentence gives them, the same population, setting and numbers.
- `p` (partial): the abstract is about the same finding but backs it only loosely: a weaker or narrower
  result, a different population or outcome, or numbers that do not match.
- `u` (unsupported): the abstract backs no claim in the sentence, or contradicts it.

Judge only from the abstract as written, not from what you know: a sentence that is true but is not stated
in the abstract is not supported. If the sentence cites several papers, look only at this one abstract.
For a pair that is hard to call, a few words after `note:` help later when reading the disagreements.

When every pair has a label, run: `uv run bioscout-eval calibrate-score evals/runs/{run_id}`
"""


def render_labeling_file(
    pair_set: PairSet, records: list[Record], abstracts: dict[str, ArticleAbstract]
) -> str:
    """The file to label by hand. It shows neither the judge's verdict nor the variant."""
    answers = {(r.item_id, r.variant): r.answer for r in records if r.answer is not None}
    parts = [
        HEADER.format(
            count=len(pair_set.pairs),
            run_id=pair_set.run_id,
            judge_sha=pair_set.judge_sha,
            seed=pair_set.seed,
        )
    ]
    for pair in pair_set.pairs:
        answer = answers[(pair.item_id, pair.variant)]
        parts.append(
            f"---\n\n## {pair.n:02d}\n\n"
            f"**Question:** {answer.question}\n\n"
            f"**Sentence:** {answer.sentences[pair.sentence].text}\n\n"
            f"**Abstract:**\n\n{format_abstracts([abstracts[pair.pmid]])}\n\n"
            "label:\nnote:\n"
        )
    return "\n".join(parts)


_SECTION = re.compile(r"^## (\d+)[ \t]*$", re.MULTILINE)
_LABEL = re.compile(r"^label:[ \t]*(.*)$", re.MULTILINE)


def read_labels(text: str) -> dict[int, Verdict | None]:
    """Labels by pair number; None where the label is still empty. Unknown labels raise ValueError."""
    labels: dict[int, Verdict | None] = {}
    bad: list[str] = []
    text = text.replace("\r\n", "\n")  # an editor on Windows may save the file with CRLF
    sections = list(_SECTION.finditer(text))
    for section, following in zip(sections, [*sections[1:], None], strict=True):
        n = int(section.group(1))
        block = text[section.end() : following.start() if following else len(text)]
        # The label line comes after the abstract, so take the last one in the block.
        values = _LABEL.findall(block)
        value = values[-1].strip().lower() if values else ""
        if not value:
            labels[n] = None
        elif value in LABELS:
            labels[n] = LABELS[value]
        else:
            bad.append(f"#{n:02d}: {value!r}")
    if bad:
        raise ValueError("Labels must be s, p or u (or the full word). Fix: " + ", ".join(bad))
    return labels


def cohen_kappa(pairs: list[tuple[str, str]]) -> float | None:
    """Agreement between two raters beyond what their label frequencies alone would give."""
    if not pairs:
        return None
    n = len(pairs)
    observed = sum(a == b for a, b in pairs) / n
    first, second = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    expected = sum(first[label] * second[label] for label in first) / n**2
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def score_labels(
    pair_set: PairSet, labels: dict[int, Verdict | None], judged: dict[Key, JudgedSentence]
) -> tuple[Calibration, list[Disagreement]]:
    """Compare hand labels with the judge's verdicts on the same pairs."""
    rated: list[tuple[Verdict, Verdict]] = []
    disagreements: list[Disagreement] = []
    for pair in pair_set.pairs:
        human = labels.get(pair.n)
        entry = judged.get((pair.item_id, pair.variant, pair.sentence))
        if human is None or entry is None or entry.judgment is None:
            continue
        judge = next(c.verdict for c in entry.judgment.citations if c.pmid == pair.pmid)
        rated.append((human, judge))
        if human is not judge:
            disagreements.append(
                Disagreement(
                    n=pair.n,
                    item_id=pair.item_id,
                    variant=pair.variant,
                    pmid=pair.pmid,
                    human=human,
                    judge=judge,
                )
            )

    result = Calibration(
        run_id=pair_set.run_id,
        judge_sha=pair_set.judge_sha,
        seed=pair_set.seed,
        pairs=len(pair_set.pairs),
        labelled=len(rated),
        confusion={h.value: {j.value: sum((h, j) == r for r in rated) for j in Verdict} for h in Verdict},
    )
    if rated:
        three = [(h.value, j.value) for h, j in rated]
        two = [(str(h is Verdict.SUPPORTED), str(j is Verdict.SUPPORTED)) for h, j in rated]
        result.agreement = sum(a == b for a, b in three) / len(three)
        result.kappa = cohen_kappa(three)
        result.agreement_supported_vs_rest = sum(a == b for a, b in two) / len(two)
        result.kappa_supported_vs_rest = cohen_kappa(two)
    return result, disagreements


def describe_calibration(result: Calibration, disagreements: list[Disagreement]) -> list[str]:
    """The result for the console."""
    lines = [f"labelled {result.labelled} of {result.pairs} pairs (judge prompt {result.judge_sha})"]
    if result.kappa is None or result.agreement is None:
        return lines
    verdict = "meets" if result.kappa >= KAPPA_TARGET else "is below"
    lines += [
        f"three verdicts: agreement {result.agreement:.1%}, Cohen's kappa {result.kappa:.2f} "
        f"({verdict} the {KAPPA_TARGET:.2f} target)",
        f"supported vs the rest: agreement {result.agreement_supported_vs_rest:.1%}, "
        f"kappa {result.kappa_supported_vs_rest:.2f}",
        "",
        "hand label (rows) by judge verdict (columns):",
        f"{'':<12}" + "".join(f"{v.value:>12}" for v in Verdict),
    ]
    lines += [
        f"{h.value:<12}" + "".join(f"{result.confusion[h.value][j.value]:>12}" for j in Verdict)
        for h in Verdict
    ]
    if disagreements:
        lines += ["", "disagreements:"]
        lines += [
            f"  #{d.n:02d} you={d.human.value} judge={d.judge.value} ({d.variant} {d.item_id}, PMID {d.pmid})"
            for d in disagreements
        ]
    return lines
