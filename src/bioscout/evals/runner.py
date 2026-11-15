"""Run every (item, variant) pair, score it, and write records, metadata and a report."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bioscout.evals.costs import render_costs, summarize_steps
from bioscout.evals.datasets import EvalItem
from bioscout.evals.judging import load_judgments, summarize_judgments
from bioscout.evals.metrics import Record, render_markdown, score_item, summarize
from bioscout.evals.variants import Context, Variant
from bioscout.verifier import JUDGE_SHA


async def answer_and_score(item: EvalItem, variant: Variant, ctx: Context) -> Record:
    record = Record(item_id=item.id, dataset=item.dataset, variant=variant.key)
    start = time.perf_counter()
    try:
        answer = await variant.run(item, ctx)
        answer.usage.latency_ms = round((time.perf_counter() - start) * 1000)
        record.scores = await score_item(item, answer, ctx.pubmed)
        record.answer = answer
    except Exception as exc:  # one bad item is recorded, it must not sink the whole run
        record.error = f"{type(exc).__name__}: {exc}"
    return record


def _write_meta(run_dir: Path, meta: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _append(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(line + "\n")


async def run_eval(
    items: list[EvalItem],
    variants: list[Variant],
    ctx: Context,
    run_dir: Path,
    meta: dict[str, Any],
    concurrency: int = 4,
    progress: Callable[[str], None] = print,
) -> list[Record]:
    """Records are appended as they finish, so an interrupted run keeps what it already paid for."""
    meta = meta | {
        "started": datetime.now(UTC).isoformat(timespec="seconds"),
        "variants": {v.key: {"name": v.name, "prompt_sha": v.prompt_sha} for v in variants},
        "items": len(items),
    }
    await asyncio.to_thread(_write_meta, run_dir, meta)

    semaphore = asyncio.Semaphore(concurrency)
    # Appends from two worker threads can land at the same offset on Windows, losing a record.
    write_lock = asyncio.Lock()
    total = len(items) * len(variants)
    done = 0
    records: list[Record] = []

    async def one(item: EvalItem, variant: Variant) -> None:
        nonlocal done
        async with semaphore:
            record = await answer_and_score(item, variant, ctx)
        records.append(record)
        async with write_lock:
            await asyncio.to_thread(_append, run_dir / "records.jsonl", record.model_dump_json())
        done += 1
        progress(f"[{done}/{total}] {_progress_line(record)}")

    await asyncio.gather(*(one(item, v) for v in variants for item in items))

    meta["finished"] = datetime.now(UTC).isoformat(timespec="seconds")
    await asyncio.to_thread(_write_meta, run_dir, meta)
    return records


def _progress_line(r: Record) -> str:
    head = f"{r.variant} {r.item_id}"
    if r.error:
        return f"{head} ERROR {r.error[:120]}"
    assert r.answer is not None and r.scores is not None
    if r.answer.refused:
        return f"{head} refused"
    usd = r.answer.usage.usd
    cost = f"${usd:.3f}" if usd is not None else "$?"
    return f"{head} cited={r.scores.cited} fabricated={r.scores.fabricated} {cost}"


def load_records(run_dir: Path) -> list[Record]:
    lines = (run_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()
    return [Record.model_validate_json(line) for line in lines if line.strip()]


def write_report(run_dir: Path, reports_dir: Path) -> Path:
    """Summarise a run's records into reports/<run_id>.json and .md (numbers only, no answer text)."""
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    records = load_records(run_dir)
    rows = summarize(records)
    judged = load_judgments(run_dir)
    by_group = summarize_judgments(records, judged)
    if any(by_group.values()):  # at least one group is judged in full with the current judge prompt
        rows = [row | by_group.get((row["variant"], row["dataset"]), {}) for row in rows]
        meta = meta | {"judge": {"prompt_sha": JUDGE_SHA, "sentences": len(judged)}}
        # The judge's agreement with hand labels, if this judge prompt has been calibrated on this run.
        calibration = reports_dir / f"{meta['run_id']}.calibration.json"
        if calibration.exists():
            result = json.loads(calibration.read_text(encoding="utf-8"))
            if result["judge_sha"] == JUDGE_SHA:
                keys = ("pairs", "agreement", "kappa", "kappa_supported_vs_rest")
                meta = meta | {"calibration": {key: result[key] for key in keys}}
    reports_dir.mkdir(parents=True, exist_ok=True)
    out = reports_dir / f"{meta['run_id']}.json"
    # Reports are committed; LF keeps them identical on Windows and in the repository.
    steps = summarize_steps(records)
    report = {"meta": meta, "rows": rows, "steps": steps}
    out.write_text(json.dumps(report, indent=2), encoding="utf-8", newline="\n")
    text = render_markdown(rows, meta) + "\n" + render_costs(rows, steps)
    out.with_suffix(".md").write_text(text, encoding="utf-8", newline="\n")
    return out
