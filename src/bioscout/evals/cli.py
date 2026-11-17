"""bioscout-eval: run the evaluation, rebuild a report, or check the datasets.

uv run bioscout-eval run --variants A,B --datasets pubmedqa,domain --limit 10
uv run bioscout-eval judge evals/runs/<run_id> --variants C --limit 10
uv run bioscout-eval calibrate-sample evals/runs/<run_id>
uv run bioscout-eval calibrate-score evals/runs/<run_id>
uv run bioscout-eval report evals/runs/<run_id>
uv run bioscout-eval check-datasets
uv run bioscout-eval gate --since HEAD^
uv run bioscout-eval readme --check
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from bioscout import tracing
from bioscout.agent.rank import default_ranker
from bioscout.config import CacheMode, Settings
from bioscout.evals import gate, readme
from bioscout.evals.calibration import (
    PairSet,
    describe_calibration,
    read_labels,
    render_labeling_file,
    sample_pairs,
    score_labels,
)
from bioscout.evals.datasets import DATASET_FILES, datasets_sha, load_dataset
from bioscout.evals.judging import describe_judgments, judge_run, load_judgments
from bioscout.evals.runner import load_records, run_eval, write_report
from bioscout.evals.selection import SELECTORS, build_agent, render_selection, select_all, summarize_selection
from bioscout.evals.variants import VARIANTS, Context
from bioscout.llm import ClaudeLLM
from bioscout.mcp_server import Sources
from bioscout.models import ArticleAbstract
from bioscout.tools import open_literature
from bioscout.verifier import JUDGE_SHA


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


def _can_call_claude(settings: Settings) -> bool:
    """Replaying from the cache needs no key; anything else does."""
    if settings.cache_mode is CacheMode.READONLY:
        return True
    if any(os.environ.get(k) for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")):
        return True
    print(
        "ANTHROPIC_API_KEY is not set. Add it to .env, or replay recorded responses with "
        "BIOSCOUT_CACHE_MODE=readonly.",
        file=sys.stderr,
    )
    return False


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
    if not _can_call_claude(settings):
        return 2
    sources = Sources(settings)
    traces = tracing.configure(settings)
    llm = tracing.traced_llm(ClaudeLLM(settings))
    variants = [VARIANTS[v] for v in args.variants]
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}-{''.join(args.variants)}"
    run_dir = args.evals_dir / "runs" / run_id
    meta = {
        "run_id": run_id,
        "model": llm.model,
        "effort": llm.effort,
        "cache_mode": str(settings.cache_mode),
        "tools": settings.tools,
        "datasets": args.datasets,
        "limit": args.limit,
        "datasets_sha": datasets_sha(items),
        "commit": _git_commit(),
    }
    print(f"run {run_id}: {len(items)} items x {len(variants)} variants, model {llm.model} ({llm.effort})")
    if traces:
        print(f"tracing to {traces}")
    try:
        async with open_literature(sources, settings.through_mcp) as pubmed:
            ctx = Context(llm, tracing.traced_literature(pubmed))
            records = await run_eval(items, variants, ctx, run_dir, meta, args.concurrency)
    finally:
        await sources.http.aclose()
    report = write_report(run_dir, args.evals_dir / "reports", args.evals_dir / "datasets")
    print(report.with_suffix(".md").read_text(encoding="utf-8"))
    print(f"records: {run_dir / 'records.jsonl'}\nreport:  {report}")
    return 1 if all(r.error for r in records) else 0


async def cmd_judge(args: argparse.Namespace) -> int:
    """Judge the citations of a finished run (all of it, or some variants and the first N items of each
    dataset), then rebuild its report with citation precision and coverage."""
    records = [r for r in load_records(args.run_dir) if not args.variants or r.variant in args.variants]
    if args.limit:
        keep: set[str] = set()
        for name in {r.dataset for r in records}:
            keep |= {i.id for i in load_dataset(name, args.evals_dir / "datasets")[: args.limit]}
        records = [r for r in records if r.item_id in keep]
    cited = sum(bool(s.citations) for r in records if r.answer for s in r.answer.sentences)

    settings = Settings.from_env()
    if not _can_call_claude(settings):
        return 2
    sources = Sources(settings)
    traces = tracing.configure(settings)
    llm = tracing.traced_llm(ClaudeLLM(settings))
    print(f"judge {args.run_dir.name}: {len(records)} answers, {cited} cited sentences, model {llm.model}")
    if traces:
        print(f"tracing to {traces}")
    try:
        with tracing.context(session_id=_run_id(args.run_dir), tags=["judge"]):
            pubmed = tracing.traced_literature(sources.pubmed)
            judged = await judge_run(records, llm, pubmed, args.run_dir, args.concurrency)
    finally:
        await sources.http.aclose()
    # What was judged just now. The report shows judge columns only for groups judged in full.
    print("\n".join(describe_judgments(records, judged)))
    report = write_report(args.run_dir, args.evals_dir / "reports", args.evals_dir / "datasets")
    print(report.with_suffix(".md").read_text(encoding="utf-8"))
    print(f"judgments: {args.run_dir / 'judgments.jsonl'}\nreport:    {report}")
    return 0 if judged or not cited else 1


def _run_id(run_dir: Path) -> str:
    return json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))["run_id"]


def _labels_path(args: argparse.Namespace) -> Path:
    return args.labels or args.evals_dir / "review" / "calibration.md"


async def cmd_calibrate_sample(args: argparse.Namespace) -> int:
    """Write the file of (sentence, cited abstract) pairs to label by hand, without the judge's verdicts."""
    judged = load_judgments(args.run_dir)
    if not judged:
        print("No verdicts from the current judge prompt. Run `bioscout-eval judge` first.", file=sys.stderr)
        return 2
    out = _labels_path(args)
    if out.exists() and not args.force:
        try:
            started = any(read_labels(out.read_text(encoding="utf-8-sig")).values())
        except ValueError:
            started = True
        if started:  # hand labels are not ours to throw away
            print(f"{out} already holds labels. Pass --force to replace it.", file=sys.stderr)
            return 2

    pairs = sample_pairs(judged, n=args.n, seed=args.seed)
    pair_set = PairSet(run_id=_run_id(args.run_dir), judge_sha=JUDGE_SHA, seed=args.seed, pairs=pairs)
    pmids = sorted({p.pmid for p in pairs})
    abstracts: dict[str, ArticleAbstract] = {}
    sources = Sources(Settings.from_env())
    try:
        for start in range(0, len(pmids), 50):
            batch = await sources.pubmed.abstracts(pmids[start : start + 50])
            abstracts |= {a.pmid: a for a in batch.articles}
    finally:
        await sources.http.aclose()

    out.parent.mkdir(parents=True, exist_ok=True)
    text = render_labeling_file(pair_set, load_records(args.run_dir), abstracts)
    out.write_text(text, encoding="utf-8", newline="\n")
    out.with_suffix(".pairs.json").write_text(
        pair_set.model_dump_json(indent=2), encoding="utf-8", newline="\n"
    )
    print(f"{len(pairs)} pairs to label: {out}")
    print(f"Fill in each `label:` with s, p or u, then run: bioscout-eval calibrate-score {args.run_dir}")
    return 0


def cmd_calibrate_score(args: argparse.Namespace) -> int:
    """Compare the hand labels with the judge. The result is saved once every pair has a label."""
    labels_path = _labels_path(args)
    pairs_path = labels_path.with_suffix(".pairs.json")
    if not labels_path.exists() or not pairs_path.exists():
        print(
            f"No labeling file at {labels_path}. Run `bioscout-eval calibrate-sample` first.", file=sys.stderr
        )
        return 2
    pair_set = PairSet.model_validate_json(pairs_path.read_text(encoding="utf-8"))
    if pair_set.run_id != _run_id(args.run_dir):
        print(f"{labels_path} was sampled from run {pair_set.run_id}, not from this run.", file=sys.stderr)
        return 2
    try:
        labels = read_labels(labels_path.read_text(encoding="utf-8-sig"))
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2

    result, disagreements = score_labels(pair_set, labels, load_judgments(args.run_dir, pair_set.judge_sha))
    print("\n".join(describe_calibration(result, disagreements)))
    if result.labelled < result.pairs:
        print(f"\n{result.pairs - result.labelled} pairs have no label yet. The result is saved once all do.")
        return 1
    reports_dir = args.evals_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out = reports_dir / f"{pair_set.run_id}.calibration.json"
    out.write_text(result.model_dump_json(indent=2), encoding="utf-8", newline="\n")
    write_report(args.run_dir, reports_dir, args.evals_dir / "datasets")
    print(f"\nsaved: {out}")
    return 0


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


async def cmd_selection(args: argparse.Namespace) -> int:
    """Run the agent up to its choice of abstracts, once per variant, and compare what was chosen. No
    answer is written. With BIOSCOUT_CACHE_MODE=readonly a variant run before is replayed at no cost."""
    unknown = [v for v in args.variants if v not in SELECTORS]
    if unknown:
        print(
            f"Unknown variant(s) {unknown}. Choices differ only between: {', '.join(SELECTORS)}",
            file=sys.stderr,
        )
        return 2
    items = []
    for name in args.datasets:
        loaded = load_dataset(name, args.evals_dir / "datasets")
        items += loaded[: args.limit] if args.limit else loaded

    settings = Settings.from_env()
    if not _can_call_claude(settings):
        return 2
    sources = Sources(settings)
    llm = ClaudeLLM(settings)
    ranker = default_ranker() if "E" in args.variants else None
    if ranker is not None:  # load the model now, so the first question's time is not the loading time
        await asyncio.to_thread(ranker.embedder.embed_query, "warm up")
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}-select-{''.join(args.variants)}"
    meta = {
        "run_id": run_id,
        "model": llm.model,
        "effort": llm.effort,
        "cache_mode": str(settings.cache_mode),
        "ranker": ranker.name if ranker else None,
        "datasets": args.datasets,
        "limit": args.limit,
        "items": len(items),
        "commit": _git_commit(),
    }
    print(
        f"selection {run_id}: {len(items)} items x {len(args.variants)} variants, cache {settings.cache_mode}"
    )
    try:
        async with open_literature(sources, settings.through_mcp) as pubmed:
            agents = {v: build_agent(v, llm, pubmed, ranker) for v in args.variants}
            selections = await select_all(items, agents, args.concurrency)
    finally:
        await sources.http.aclose()
    rows = summarize_selection(selections, {item.id: item for item in items})
    text = render_selection(rows, meta)
    reports = args.evals_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    out = reports / f"{run_id}.json"
    out.write_text(json.dumps({"meta": meta, "rows": rows}, indent=2), encoding="utf-8", newline="\n")
    out.with_suffix(".md").write_text(text, encoding="utf-8", newline="\n")
    print(text)
    for failed in [s for s in selections if s.error][:5]:
        print(f"error {failed.variant} {failed.item_id}: {failed.error[:160]}")
    print(f"report: {out}")
    return 0 if any(not s.error for s in selections) else 1


def cmd_gate(args: argparse.Namespace) -> int:
    """Check the committed baseline report against the code and against the baseline of an earlier commit."""
    path = args.baseline or args.evals_dir / "baseline.json"
    baseline = gate.Baseline.load(path.read_text(encoding="utf-8"), Path.cwd())
    previous = None
    if args.since:
        shown = subprocess.run(
            ["git", "show", f"{args.since}:{path.as_posix()}"],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        if shown.returncode == 0:  # that commit may predate the baseline, or not be in this checkout
            previous = gate.Baseline.load(shown.stdout, Path.cwd())
    checks = gate.run_gate(baseline, gate.Code.current(args.evals_dir / "datasets"), previous)
    print(gate.render(baseline, checks))
    return 0 if gate.passed(checks) else 1


def cmd_readme(args: argparse.Namespace) -> int:
    """Write the README's results block and its charts from the baseline report, or only compare them."""
    path = args.baseline or args.evals_dir / "baseline.json"
    baseline = gate.Baseline.load(path.read_text(encoding="utf-8"), Path.cwd())
    if args.check:
        stale = readme.stale(Path.cwd(), baseline)
        for name in stale:
            print(f"out of date: {name}")
        if stale:
            print(f"These no longer show {baseline.report_path}. Run `uv run bioscout-eval readme`.")
        else:
            print(f"README.md and its charts show {baseline.report_path}")
        return 1 if stale else 0
    changed = readme.write(Path.cwd(), baseline)
    for name in changed:
        print(f"wrote {name}")
    if not changed:
        print(f"README.md and its charts already show {baseline.report_path}")
    return 0


def main() -> None:
    # A redirected stdout on Windows uses the locale code page (cp949 on a Korean PC), which cannot encode
    # the "—" of an empty table cell. Replace such characters rather than crash after the work is done.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(prog="bioscout-eval")
    parser.add_argument("--evals-dir", type=Path, default=Path("evals"))
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="answer and score items, then write a report")
    run.add_argument("--variants", type=_csv, default=["A", "B"])
    run.add_argument("--datasets", type=_csv, default=["pubmedqa", "domain"])
    run.add_argument("--limit", type=int, default=0, help="first N items of each dataset (0 = all)")
    run.add_argument("--concurrency", type=int, default=4)

    judge = sub.add_parser("judge", help="judge the citations of an existing run, then rebuild its report")
    judge.add_argument("run_dir", type=Path)
    judge.add_argument("--variants", type=_csv, default=[], help="only these variants (default: all)")
    judge.add_argument("--limit", type=int, default=0, help="first N items of each dataset (0 = all)")
    judge.add_argument("--concurrency", type=int, default=4)

    sample = sub.add_parser("calibrate-sample", help="write pairs from a judged run to label by hand")
    sample.add_argument("run_dir", type=Path)
    sample.add_argument("--n", type=int, default=60, help="number of (sentence, abstract) pairs")
    sample.add_argument("--seed", type=int, default=2026)
    sample.add_argument("--labels", type=Path, help="labeling file (default: evals/review/calibration.md)")
    sample.add_argument("--force", action="store_true", help="replace a labeling file that holds labels")

    score = sub.add_parser("calibrate-score", help="compare the hand labels with the judge (Cohen's kappa)")
    score.add_argument("run_dir", type=Path)
    score.add_argument("--labels", type=Path, help="labeling file (default: evals/review/calibration.md)")

    report = sub.add_parser("report", help="rebuild the report of an existing run")
    report.add_argument("run_dir", type=Path)

    sub.add_parser("check-datasets", help="verify that every dataset PMID exists in PubMed")

    gate_cmd = sub.add_parser(
        "gate", help="check that the baseline report matches the code and did not regress"
    )
    gate_cmd.add_argument("--baseline", type=Path, help="default: evals/baseline.json")
    gate_cmd.add_argument("--since", help="git revision whose baseline to compare with, e.g. HEAD^")

    readme_cmd = sub.add_parser(
        "readme", help="write the README's results chart and table from the baseline report"
    )
    readme_cmd.add_argument("--baseline", type=Path, help="default: evals/baseline.json")
    readme_cmd.add_argument("--check", action="store_true", help="write nothing; exit 1 when out of date")

    selection = sub.add_parser("selection", help="compare how variants choose abstracts, writing no answer")
    selection.add_argument("--variants", type=_csv, default=["C", "E"])
    selection.add_argument("--datasets", type=_csv, default=["pubmedqa", "domain"])
    selection.add_argument("--limit", type=int, default=0, help="first N items of each dataset (0 = all)")
    selection.add_argument("--concurrency", type=int, default=4)

    retrieval = sub.add_parser("retrieval", help="PubMedQA source-paper recall@10 of verbatim search, no LLM")
    retrieval.add_argument("--limit", type=int, default=0)

    args = parser.parse_args()
    if args.command == "run":
        sys.exit(asyncio.run(cmd_run(args)))
    if args.command == "judge":
        sys.exit(asyncio.run(cmd_judge(args)))
    if args.command == "calibrate-sample":
        sys.exit(asyncio.run(cmd_calibrate_sample(args)))
    if args.command == "calibrate-score":
        sys.exit(cmd_calibrate_score(args))
    if args.command == "report":
        out = write_report(args.run_dir, args.evals_dir / "reports", args.evals_dir / "datasets")
        print(out.with_suffix(".md").read_text(encoding="utf-8"))
        sys.exit(0)
    if args.command == "retrieval":
        sys.exit(asyncio.run(cmd_retrieval(args)))
    if args.command == "gate":
        sys.exit(cmd_gate(args))
    if args.command == "readme":
        sys.exit(cmd_readme(args))
    if args.command == "selection":
        sys.exit(asyncio.run(cmd_selection(args)))
    sys.exit(asyncio.run(cmd_check(args)))


if __name__ == "__main__":
    main()
