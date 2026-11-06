"""Prompts and schemas for the agent's own steps. The answer step uses bioscout.prompts, shared with B."""

from __future__ import annotations

from typing import Any

PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "searches": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "purpose": {"type": "string"},
                    "query": {"type": "string"},
                },
                "required": ["purpose", "query"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["searches"],
    "additionalProperties": False,
}

SYSTEM_PLAN = """\
You plan PubMed searches for a biomedical research question.
First identify the question's concepts: the population or condition, the intervention or exposure, \
the outcome, and the comparison if there is one.
Then write 2 to 4 PubMed queries. Within a concept, join synonyms with OR, pairing a MeSH heading \
("Term"[MeSH]) with free-text synonyms ("term"[tiab]); join concepts with AND. Use at most three \
AND-ed concepts per query, because PubMed returns nothing for over-constrained queries.
Make one query broader than the others so it can also find evidence against the expected answer.
Do not add date limits or publication-type filters. Give each query a short purpose: the part of the \
question it looks for."""

SCREEN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "relevant_pmids": {"type": "array", "items": {"type": "string"}},
        "enough": {"type": "boolean"},
        "missing": {"type": "string"},
        "next_queries": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["relevant_pmids", "enough", "missing", "next_queries"],
    "additionalProperties": False,
}

SYSTEM_SCREEN = """\
You screen PubMed results for a biomedical research question. You get the question, the searches run \
so far with their hit counts, and candidate abstracts.
In "relevant_pmids", list the candidates whose abstract reports findings that help answer the question \
(not just the same topic), most useful first, at most 10. Keep papers whose findings go against the \
likely answer.
Set "enough" to true when those abstracts are enough to answer the question. Otherwise say in \
"missing" what evidence is missing, and give up to 3 new PubMed queries in "next_queries": broader ones \
if the searches found few hits, different terms if the results were off-topic. Never repeat a query \
that was already run."""
