"""Build evals/datasets/pubmedqa_100.jsonl: a fixed random sample of the expert-labelled PubMedQA set.

    uv run python scripts/build_pubmedqa.py

Only the PMID, question and yes/no/maybe label are kept. The source file's CONTEXTS (abstract text) and
LONG_ANSWER (the abstract's conclusion) are dropped: this repo never stores abstracts.
Re-running with the same seed reproduces the same 100 questions as long as the source file is unchanged
(its SHA-256 is printed and recorded in evals/datasets/README.md).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import httpx

SOURCE = "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json"
OUT = Path(__file__).resolve().parents[1] / "evals" / "datasets" / "pubmedqa_100.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    raw = httpx.get(SOURCE, timeout=60, follow_redirects=True).raise_for_status().content
    data: dict[str, dict] = json.loads(raw)
    pmids = sorted(data)
    sample = sorted(random.Random(args.seed).sample(pmids, args.n), key=int)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="\n") as f:
        for pmid in sample:
            row = data[pmid]
            item = {
                "id": f"pqa-{pmid}",
                "dataset": "pubmedqa",
                "question": row["QUESTION"].strip(),
                "answer": row["final_decision"],
                "pmid": pmid,
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    labels = Counter(data[p]["final_decision"] for p in sample)
    print(f"source sha256 {hashlib.sha256(raw).hexdigest()}  ({len(pmids)} labelled questions)")
    print(f"wrote {len(sample)} items to {OUT}  labels: {dict(sorted(labels.items()))}")


if __name__ == "__main__":
    main()
