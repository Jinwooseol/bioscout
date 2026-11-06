"""The answer schema and prompts shared by every answering step: eval variants A to C and the demo.

B and C answer with the same system prompt, schema and abstract format, so a difference between them
comes from which abstracts they read, not from how the answer was asked for.
"""

from __future__ import annotations

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
    sentences = [
        CitedSentence(text=s["text"].strip(), citations=[str(c).strip() for c in s["citations"]])
        for s in result.data["sentences"]
    ]
    return base.model_copy(update={"decision": result.data["decision"], "sentences": sentences})
