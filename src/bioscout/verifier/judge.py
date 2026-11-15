"""The citation judge: do the PubMed abstracts a sentence cites back what the sentence says?

One LLM call per sentence, with every abstract that sentence cites. The eval turns the verdicts into
citation precision and coverage; variant D's verify step will run the same judge before an answer is shown.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from pydantic import BaseModel, Field

from bioscout.answer import Usage, Verdict
from bioscout.llm import LLM
from bioscout.models import ArticleAbstract
from bioscout.prompts import format_abstracts
from bioscout.sources.pubmed import normalize_pmids

_VERDICTS = [v.value for v in Verdict]

JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "pmid": {"type": "string"},
                    "verdict": {"type": "string", "enum": _VERDICTS},
                    "evidence": {"type": "string"},
                },
                "required": ["pmid", "verdict", "evidence"],
                "additionalProperties": False,
            },
        },
        "sentence_verdict": {"type": "string", "enum": _VERDICTS},
        "missing": {"type": "string"},
    },
    "required": ["citations", "sentence_verdict", "missing"],
    "additionalProperties": False,
}

SYSTEM_JUDGE = """\
You check citations in an answer to a biomedical research question. You get the question, one sentence \
from the answer, and the PubMed abstracts that the sentence cites.
Judge only from the abstracts as given. Do not use what you know about these papers or the field: a \
sentence that is true but is not stated in the abstracts is not supported.

For each abstract, set "verdict":
- supported: the abstract clearly states at least one factual claim that the sentence makes, with the \
same direction of effect and, where the sentence gives them, the same population, setting and numbers.
- partial: the abstract is about the same finding but backs it only loosely: a weaker or narrower \
result, a different population or outcome, or numbers that do not match.
- unsupported: the abstract backs no claim in the sentence, or contradicts it.
In "evidence", copy word for word the one passage of that abstract, at most 40 words, that the verdict \
rests on. Leave it empty when nothing in the abstract is relevant.

Then set "sentence_verdict" for the abstracts taken together:
- supported: every factual claim in the sentence is clearly stated in at least one abstract.
- partial: some claims are backed and others are not, or the backing is only loose.
- unsupported: no claim is backed.
In "missing", name the part of the sentence that no abstract backs, or leave it empty."""

# Recorded with every judgment: verdicts from different judge prompts are not comparable.
JUDGE_SHA = hashlib.sha256(json.dumps([SYSTEM_JUDGE, JUDGE_SCHEMA], sort_keys=True).encode()).hexdigest()[:12]

_RANK = {Verdict.UNSUPPORTED: 0, Verdict.PARTIAL: 1, Verdict.SUPPORTED: 2}
_BY_RANK = {rank: verdict for verdict, rank in _RANK.items()}


class JudgeRefusedError(RuntimeError):
    """The judge call was refused, so the sentence has no verdict. `usage` is what the call still cost."""

    def __init__(self, message: str, usage: Usage | None = None) -> None:
        super().__init__(message)
        self.usage = usage or Usage()


class CitationJudgment(BaseModel):
    pmid: str
    verdict: Verdict
    evidence: str = Field(default="", description="Passage of the abstract the verdict rests on")
    quote_found: bool | None = Field(
        default=None, description="The evidence is word for word in the abstract (None: no evidence given)"
    )
    reason: str | None = Field(
        default=None, description="Why the LLM gave no verdict: pmid_not_found | no_abstract | not_judged"
    )


class SentenceJudgment(BaseModel):
    verdict: Verdict = Field(description="Every cited abstract taken together")
    missing: str = Field(default="", description="The part of the sentence that no abstract backs")
    citations: list[CitationJudgment] = Field(default_factory=list)


def _words(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def quote_in_abstract(evidence: str, article: ArticleAbstract) -> bool:
    """Is the quoted passage really in the abstract? Case, punctuation and spacing are ignored. The judge
    often stitches passages together with "..."; such a quote counts when every piece is in the abstract."""
    text = _words(f"{article.title} {article.abstract_text}")
    pieces = [piece for part in re.split(r"\.{3}|…", evidence) if (piece := _words(part))]
    return bool(pieces) and all(piece in text for piece in pieces)


def bound_sentence_verdict(claimed: Verdict, citations: list[CitationJudgment]) -> Verdict:
    """A sentence cannot be better backed than its best citation, and a clearly supporting citation
    backs at least part of it."""
    best = max((_RANK[c.verdict] for c in citations), default=0)
    rank = min(_RANK[claimed], best)
    if best == _RANK[Verdict.SUPPORTED]:
        rank = max(rank, _RANK[Verdict.PARTIAL])
    return _BY_RANK[rank]


async def judge_sentence(
    llm: LLM, question: str, text: str, citations: list[str], abstracts: dict[str, ArticleAbstract]
) -> tuple[SentenceJudgment, Usage]:
    """Judge one sentence against the abstracts of its citations. `abstracts` is keyed by PMID and holds
    whatever PubMed returned for them; a citation without a usable abstract is unsupported with no LLM call.
    """
    judged: dict[str, CitationJudgment] = {}
    usable: list[ArticleAbstract] = []
    for raw in dict.fromkeys(c.strip() for c in citations if c.strip()):
        pmid = next(iter(normalize_pmids([raw])), raw)
        if pmid in judged:
            continue
        article = abstracts.get(pmid)
        if article is None or not article.abstract_text:
            reason = "pmid_not_found" if article is None else "no_abstract"
            judged[pmid] = CitationJudgment(pmid=pmid, verdict=Verdict.UNSUPPORTED, reason=reason)
        else:
            judged[pmid] = CitationJudgment(pmid=pmid, verdict=Verdict.UNSUPPORTED, reason="not_judged")
            usable.append(article)
    if not usable:
        return SentenceJudgment(verdict=Verdict.UNSUPPORTED, citations=list(judged.values())), Usage()

    user = f"Question: {question}\n\nSentence: {text}\n\nCited abstracts:\n\n{format_abstracts(usable)}"
    result = await llm.json(system=SYSTEM_JUDGE, user=user, schema=JUDGE_SCHEMA)
    if result.refused or result.data is None:
        raise JudgeRefusedError(
            f"judge refused ({result.refusal_category or 'no category'})", usage=result.usage
        )

    by_pmid = {a.pmid: a for a in usable}
    for item in result.data["citations"]:
        pmid = str(item["pmid"]).strip()
        if pmid not in by_pmid:  # a verdict about an abstract that was not given
            continue
        evidence = item["evidence"].strip()
        judged[pmid] = CitationJudgment(
            pmid=pmid,
            verdict=Verdict(item["verdict"]),
            evidence=evidence,
            quote_found=quote_in_abstract(evidence, by_pmid[pmid]) if evidence else None,
        )
    verdicts = list(judged.values())
    judgment = SentenceJudgment(
        verdict=bound_sentence_verdict(Verdict(result.data["sentence_verdict"]), verdicts),
        missing=result.data["missing"].strip(),
        citations=verdicts,
    )
    return judgment, result.usage
