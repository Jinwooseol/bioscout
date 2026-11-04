"""Eval items and the JSONL files they live in (evals/datasets/)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

DATASET_FILES = {
    "pubmedqa": "pubmedqa_100.jsonl",
    "domain": "domain.jsonl",
}


class KeyPoint(BaseModel):
    point: str
    pmids: list[str] = Field(default_factory=list, description="must_cite PMIDs that support the point")


class EvalItem(BaseModel):
    id: str
    dataset: str
    question: str
    # PubMedQA: the expert yes/no/maybe label and the paper the question was written from.
    answer: Literal["yes", "no", "maybe"] | None = None
    pmid: str | None = None
    # Domain set: what a good answer cites and covers.
    topic: str | None = None
    must_cite: list[str] = Field(default_factory=list)
    key_points: list[KeyPoint] = Field(default_factory=list)
    status: str | None = None


def load_dataset(name: str, datasets_dir: Path) -> list[EvalItem]:
    path = datasets_dir / DATASET_FILES.get(name, f"{name}.jsonl")
    items = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        row.setdefault("dataset", name)
        try:
            items.append(EvalItem.model_validate(row))
        except ValueError as exc:
            raise ValueError(f"{path}:{n}: {exc}") from exc
    ids = [i.id for i in items]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate item ids")
    return items
