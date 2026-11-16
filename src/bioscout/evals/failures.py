"""Where answers go wrong, read from a run's records. Nothing is called and nothing is judged again.

Two questions. For the paper an answer should have cited (the source paper of a PubMedQA question, each
must-cite paper of a domain question): at which step was it lost? And for a wrong yes/no/maybe answer:
what kind of wrong was it? An answer that says "maybe" where the label says yes is a different failure
from one that says "no", and accuracy alone counts them the same.
"""

from __future__ import annotations

from typing import Any

from bioscout.answer import CitedAnswer
from bioscout.evals.datasets import EvalItem
from bioscout.evals.metrics import Record, format_cell

# How far a paper got, from best to worst. A paper is counted at the first step that dropped it.
PAPER_STAGES = [
    ("cited", "Cited"),
    ("not_cited", "Read, not cited"),
    ("not_selected", "Found, not selected"),
    ("not_found", "Not found by a search"),
]
WRONG_KINDS = [
    ("opposite", "Opposite (yes for no)"),
    ("hedged", "Hedged (maybe for yes/no)"),
    ("committed", "Committed (yes/no for maybe)"),
    ("no_answer", "No answer"),
]
DECISIONS = ("yes", "no", "maybe")


def paper_stage(answer: CitedAnswer, pmid: str) -> str:
    """Was the paper cited, or only given to the answer step, or only returned by a search, or never seen?"""
    if pmid in answer.cited_pmids:
        return "cited"
    if pmid in answer.context_pmids:
        return "not_cited"
    if any(pmid in search.pmids for search in answer.searches):
        return "not_selected"
    return "not_found"


def wrong_kind(label: str, decision: str | None) -> str | None:
    """The kind of mistake, or None when the answer matches the label."""
    if decision == label:
        return None
    if decision not in DECISIONS:
        return "no_answer"
    if decision == "maybe":
        return "hedged"
    return "opposite" if label != "maybe" else "committed"


def summarize_failures(records: list[Record], items: dict[str, EvalItem]) -> dict[str, list[dict[str, Any]]]:
    """Per (variant, dataset): where the expected papers were lost, and the kinds of wrong answer."""
    groups: dict[tuple[str, str], list[tuple[EvalItem, CitedAnswer]]] = {}
    for record in records:
        item = items.get(record.item_id)
        if item is not None and record.answer is not None:
            groups.setdefault((record.variant, record.dataset), []).append((item, record.answer))

    papers: list[dict[str, Any]] = []
    answers: list[dict[str, Any]] = []
    for (variant, dataset), pairs in sorted(groups.items()):
        searches = any(answer.searches for _, answer in pairs)
        stages = [
            paper_stage(answer, pmid)
            for item, answer in pairs
            for pmid in ([item.pmid] if item.pmid else item.must_cite)
        ]
        if stages:
            row: dict[str, Any] = {"variant": variant, "dataset": dataset, "papers": len(stages)}
            for key, _ in PAPER_STAGES:
                # A variant that never searches cites from memory or not at all; the steps do not apply.
                row[key] = stages.count(key) if searches or key == "cited" else None
            papers.append(row)

        labelled = [(item, answer) for item, answer in pairs if item.answer]
        if labelled:
            kinds = [
                wrong_kind(item.answer, None if answer.refused else answer.decision)
                for item, answer in labelled
            ]
            row = {"variant": variant, "dataset": dataset, "n": len(labelled)}
            row["wrong"] = sum(kind is not None for kind in kinds)
            for key, _ in WRONG_KINDS:
                row[key] = kinds.count(key)
            # Wrong although the answer step was given the question's own source paper.
            row["wrong_with_source"] = (
                sum(
                    kind is not None and bool(item.pmid) and item.pmid in answer.context_pmids
                    for kind, (item, answer) in zip(kinds, labelled, strict=True)
                )
                if searches
                else None
            )
            answers.append(row)
    return {"papers": papers, "answers": answers}


def _table(columns: list[tuple[str, str]], rows: list[dict[str, Any]]) -> list[str]:
    head = "| " + " | ".join(label for _, label in columns) + " |"
    rule = "|" + "|".join(" --- " for _ in columns) + "|"
    body = ["| " + " | ".join(format_cell(row.get(key), "str") for key, _ in columns) + " |" for row in rows]
    return [head, rule, *body]


def render_failures(failures: dict[str, list[dict[str, Any]]]) -> str:
    if not failures["papers"] and not failures["answers"]:
        return ""
    lines = ["## Where answers go wrong", ""]
    if failures["papers"]:
        lines += [
            "The paper each answer should have cited (the source paper of a PubMedQA question, every "
            "must-cite paper of a domain question), counted at the first step that lost it. A variant that "
            "does not "
            "search has no steps to lose it at.",
            "",
            *_table([("variant", "Variant"), ("dataset", "Dataset"), ("papers", "Papers"), *PAPER_STAGES],
                    failures["papers"]),
            "",
        ]  # fmt: skip
    if failures["answers"]:
        lines += [
            "Wrong yes/no/maybe answers by kind. The last column counts wrong answers that were written with "
            "the question's own source paper among the abstracts.",
            "",
            *_table([("variant", "Variant"), ("dataset", "Dataset"), ("n", "n"), ("wrong", "Wrong"),
                     *WRONG_KINDS,
                     ("wrong_with_source", "Wrong with the source paper read")], failures["answers"]),
            "",
        ]  # fmt: skip
    return "\n".join(lines)
