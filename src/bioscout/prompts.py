"""The answer schema and prompts shared by every answering step: eval variants A to C and the demo.

B and C answer with the same system prompt, schema and abstract format, so a difference between them
comes from which abstracts they read, not from how the answer was asked for.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from bioscout.answer import CitedAnswer, CitedSentence, SearchLog, Usage
from bioscout.llm import LLMResult
from bioscout.models import ArticleAbstract

ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["yes", "no", "maybe", "not_applicable"]},
        "sentences": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "citations": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["text", "citations"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["decision", "sentences"],
    "additionalProperties": False,
}

ANSWER_RULES = """\
Write 2 to 6 sentences in English. Each sentence states one point and lists in "citations" the PubMed \
IDs (PMIDs, digits only) of the papers that support it.
Set "decision" to yes, no or maybe when the question can be answered that way, otherwise not_applicable.
The answer supports reading the research literature; it is not advice about an individual patient."""

SYSTEM_GROUNDED = f"""\
You are a biomedical research assistant. Answer using only the PubMed abstracts given with the question.
{ANSWER_RULES}
Cite only PMIDs of the given abstracts, and only where the abstract supports the sentence. If the \
abstracts do not answer the question, say so and leave "citations" empty."""


def format_abstracts(articles: list[ArticleAbstract]) -> str:
    return "\n\n".join(
        f"PMID {a.pmid} | {a.title} | {a.journal} {a.year or ''}\n{a.abstract_text or '(no abstract)'}"
        for a in articles
    )


# Part of the fingerprint of every variant that answers from abstracts: change it when the rule changes.
GROUNDING = "citations limited to the abstracts given, v1"


def ground_citations(citations: Sequence[str], context: Sequence[str]) -> tuple[list[str], list[str]]:
    """Keep only citations of the abstracts the answer was written from.

    An entry that is not one of those PMIDs is searched for them: "24, 41022185" yields 41022185 when
    that paper was given, and the stray "24" is not kept. Returns the citations to keep, without
    repeats, and the entries as written that were reduced or removed."""
    allowed = set(context)
    kept: list[str] = []
    odd: list[str] = []
    for raw in citations:
        entry = str(raw).strip()
        if entry in allowed:
            found = [entry]
        else:
            odd.append(entry)
            found = [token for token in re.findall(r"\d+", entry) if token in allowed]
        kept += [pmid for pmid in found if pmid not in kept]
    return kept, odd


def answer_from_result(
    question: str,
    result: LLMResult,
    *,
    usage: Usage,
    searches: list[SearchLog] | None = None,
    context_pmids: list[str] | None = None,
) -> CitedAnswer:
    """Turn an ANSWER_SCHEMA reply into a CitedAnswer. `usage` is the run's total, this call included."""
    base = CitedAnswer(
        question=question, searches=searches or [], context_pmids=context_pmids or [], usage=usage
    )
    if result.refused or result.data is None:
        return base.model_copy(update={"refused": True})
    sentences: list[CitedSentence] = []
    ungrounded: list[str] = []
    for s in result.data["sentences"]:
        citations = [str(c).strip() for c in s["citations"]]
        if context_pmids is not None:  # written from these abstracts, so no other paper may be cited
            citations, odd = ground_citations(citations, context_pmids)
            ungrounded += odd
        sentences.append(CitedSentence(text=s["text"].strip(), citations=citations))
    update = {"decision": result.data["decision"], "sentences": sentences, "ungrounded": ungrounded}
    return base.model_copy(update=update)
