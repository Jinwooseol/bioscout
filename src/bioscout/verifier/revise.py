"""What variant D does with the judge's verdicts: keep only the citations that back a sentence, and give
a sentence that is not fully backed one chance to be rewritten from the abstracts the answer was written
from. The agent graph runs these steps; nothing here searches or decides what to drop.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from bioscout.answer import CitedSentence, Usage, Verdict
from bioscout.llm import LLM
from bioscout.models import ArticleAbstract
from bioscout.prompts import format_abstracts
from bioscout.verifier.judge import JUDGE_SCHEMA, SYSTEM_JUDGE, SentenceJudgment

REVISE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "sentences": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "text": {"type": "string"},
                    "citations": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["index", "text", "citations"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["sentences"],
    "additionalProperties": False,
}

SYSTEM_REVISE = """\
You repair sentences in an answer to a biomedical research question. A checker compared each sentence \
with the PubMed abstracts it cites and found that the abstracts do not fully back it. You get the \
question, the abstracts the answer was written from, and the sentences to repair with what the checker \
found unbacked.
For each sentence, return a version in which every factual claim is clearly stated in an abstract it \
cites: remove or narrow the unbacked part, use the abstract's own population, setting and numbers, and \
cite the PMIDs (digits only) of the abstracts that state it. Keep the sentence to the point it was \
making and do not add new claims.
If the abstracts back nothing of what a sentence says, return an empty "text" for it and the sentence \
will be removed.
Return one entry per sentence, with the "index" it was given."""


@dataclass(frozen=True)
class VerifyConfig:
    max_revisions: int = (
        1  # rounds of rewriting; a sentence still not backed after them is not rewritten again
    )


def verify_fingerprint(config: VerifyConfig | None = None) -> tuple[Any, ...]:
    """The prompts and settings of the verify step; part of variant D's prompt_sha."""
    return (SYSTEM_JUDGE, JUDGE_SCHEMA, SYSTEM_REVISE, REVISE_SCHEMA, asdict(config or VerifyConfig()))


def apply_judgment(sentence: CitedSentence, judgment: SentenceJudgment) -> CitedSentence:
    """Set the sentence's verdict and keep only the citations that carry it: the supporting ones, or
    failing that the partly supporting ones. A sentence that nothing backs keeps what it cited, so the
    record of a dropped sentence still shows it."""
    supporting = [c for c in judgment.citations if c.verdict is Verdict.SUPPORTED]
    kept = supporting or [c for c in judgment.citations if c.verdict is Verdict.PARTIAL]
    return sentence.model_copy(
        update={
            "verdict": judgment.verdict,
            "evidence": next((c.evidence for c in kept if c.evidence), None),
            "citations": [c.pmid for c in kept] if kept else sentence.citations,
        }
    )


def unbacked_note(sentence: CitedSentence, judgment: SentenceJudgment) -> str:
    """What the reviser is told about a sentence the judge did not find fully backed."""
    if not sentence.citations:
        return "The sentence cites no abstract."
    return judgment.missing or "The cited abstracts do not back it."


async def revise_sentences(
    llm: LLM,
    question: str,
    sentences: dict[int, CitedSentence],
    notes: dict[int, str],
    context: list[ArticleAbstract],
) -> tuple[dict[int, CitedSentence | None], Usage]:
    """Ask for a backed version of each sentence. Returns, by index, the rewritten sentence (not yet
    judged) or None to drop it; an index the model skipped, or everything after a refusal, is absent.
    """
    listed = "\n\n".join(
        f"[{i}] {s.text}\nCited: {', '.join(s.citations) or 'nothing'}\nNot backed: {notes[i]}"
        for i, s in sentences.items()
    )
    user = (
        f"Question: {question}\n\n"
        f"Abstracts:\n\n{format_abstracts(context) or '(None.)'}\n\n"
        f"Sentences to repair:\n\n{listed}"
    )
    result = await llm.json(system=SYSTEM_REVISE, user=user, schema=REVISE_SCHEMA)
    if result.refused or result.data is None:
        return {}, result.usage

    given = {a.pmid for a in context}
    revised: dict[int, CitedSentence | None] = {}
    for item in result.data["sentences"]:
        index = item["index"]
        if index not in sentences or index in revised:
            continue
        text = item["text"].strip()
        # Only abstracts the reviser was shown can be cited.
        citations = [p for p in dict.fromkeys(str(c).strip() for c in item["citations"]) if p in given]
        revised[index] = CitedSentence(text=text, citations=citations) if text else None
    return revised, result.usage
