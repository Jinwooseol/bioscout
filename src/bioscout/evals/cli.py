"""bioscout-eval: run the evaluation, rebuild a report, or check the datasets.

uv run bioscout-eval run --variants A,B --datasets pubmedqa,domain --limit 10
uv run bioscout-eval report evals/runs/<run_id>
uv run bioscout-eval check-datasets
"""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from bioscout.config import CacheMode, Settings
from bioscout.evals.datasets import DATASET_FILES, load_dataset
from bioscout.evals.runner import run_eval, write_report
from bioscout.evals.variants import VARIANTS, Context
from bioscout.llm import ClaudeLLM
from bioscout.mcp_server import Sources


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None


def _csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


async def cmd_run(args: argparse.Namespace) -> int:
    unknown = [v for v in args.variants if v not in VARIANTS]
    if unknown:
        print(f"Unknown variant(s) {unknown}. Available: {', '.join(VARIANTS)}", file=sys.stderr)
        return 2
    datasets_dir = args.evals_dir / "datasets"
    items = []
    for name in args.datasets:
        loaded = load_dataset(name, datasets_dir)
        items += loaded[: args.limit] if args.limit else loaded

    settings = Settings.from_env()
    has_credentials = any(os.environ.get(k) for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"))
    if settings.cache_mode is not CacheMode.READONLY and not has_credentials:
        print(
            "ANTHROPIC_API_KEY is not set. Add it to .env, or replay recorded responses with "
            "BIOSCOUT_CACHE_MODE=readonly.",
            file=sys.stderr,
        )
        return 2
    sources = Sources(settings)
    llm = ClaudeLLM(settings)
    variants = [VARIANTS[v] for v in args.variants]
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}-{''.join(args.variants)}"
    run_dir = args.evals_dir / "runs" / run_id
    meta = {
        "run_id": run_id,
        "model": llm.model,
        "effort": llm.effort,
        "cache_mode": str(settings.cache_mode),
        "datasets": args.datasets,
        "limit": args.limit,
        "commit": _git_commit(),
    }
    print(f"run {run_id}: {len(items)} items x {len(variants)} variants, model {llm.model} ({llm.effort})")
    try:
        records = await run_eval(
            items, variants, Context(llm, sources.pubmed), run_dir, meta, args.concurrency
        )
    finally:
        await sources.http.aclose()
    report = write_report(run_dir, args.evals_dir / "reports")
    print(report.with_suffix(".md").read_text(encoding="utf-8"))
    print(f"records: {run_dir / 'records.jsonl'}\nreport:  {report}")
    return 1 if all(r.error for r in records) else 0


async def cmd_check(args: argparse.Namespace) -> int:
    """Every PMID a dataset relies on must exist in PubMed (the domain set is written by hand)."""
    settings = Settings.from_env()
    sources = Sources(settings)
    failures = 0
    try:
        for name in DATASET_FILES:
            path = args.evals_dir / "datasets" / DATASET_FILES[name]
            if not path.exists():
                print(f"{name}: missing {path}")
                continue
            items = load_dataset(name, args.evals_dir / "datasets")
            pmids = sorted({p for i in items for p in [*(i.must_cite), *([i.pmid] if i.pmid else [])]})
            missing: list[str] = []
            for start in range(0, len(pmids), 200):
                missing += (await sources.pubmed.check_pmids(pmids[start : start + 200])).missing
            bad_points = [i.id for i in items for k in i.key_points if not set(k.pmids) <= set(i.must_cite)]
            failures += bool(missing) + bool(bad_points)
            print(f"{name}: {len(items)} items, {len(pmids)} PMIDs, missing={missing or 'none'}", end="")
            print(f", key points citing non-must-cite PMIDs: {bad_points}" if bad_points else "")
    finally:
        await sources.http.aclose()
    return 1 if failures else 0


async def cmd_retrieval(args: argparse.Namespace) -> int:
    """Retrieval only, no LLM: is each PubMedQA source paper in the top 10 for its question as a query?"""
    items = load_dataset("pubmedqa", args.evals_dir / "datasets")
    items = items[: args.limit] if args.limit else items
    sources = Sources(Settings.from_env())
    try:
        results = await asyncio.gather(*(sources.pubmed.search(i.question, max_results=10) for i in items))
    finally:
        await sources.http.aclose()
    found = sum(i.pmid in [a.pmid for a in r.articles] for i, r in zip(items, results, strict=True))
    empty = sum(not r.articles for r in results)
    print(
        f"verbatim question as query, top 10: source paper found for {found}/{len(items)} "
        f"({found / len(items):.0%}); {empty} searches returned nothing"
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(prog="bioscout-eval")
    parser.add_argument("--evals-dir", type=Path, default=Path("evals"))
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="answer and score items, then write a report")
    run.add_argument("--variants", type=_csv, default=["A", "B"])
    run.add_argument("--datasets", type=_csv, default=["pubmedqa", "domain"])
    run.add_argument("--limit", type=int, default=0, help="first N items of each dataset (0 = all)")
    run.add_argument("--concurrency", type=int, default=4)

    report = sub.add_parser("report", help="rebuild the report of an existing run")
    report.add_argument("run_dir", type=Path)

    sub.add_parser("check-datasets", help="verify that every dataset PMID exists in PubMed")

    retrieval = sub.add_parser("retrieval", help="PubMedQA source-paper recall@10 of verbatim search, no LLM")
    retrieval.add_argument("--limit", type=int, default=0)

    args = parser.parse_args()
    if args.command == "run":
        sys.exit(asyncio.run(cmd_run(args)))
    if args.command == "report":
        out = write_report(args.run_dir, args.evals_dir / "reports")
        print(out.with_suffix(".md").read_text(encoding="utf-8"))
        sys.exit(0)
    if args.command == "retrieval":
        sys.exit(asyncio.run(cmd_retrieval(args)))
    sys.exit(asyncio.run(cmd_check(args)))


if __name__ == "__main__":
    main()
