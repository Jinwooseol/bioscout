"""Cost and latency tables of a run: per variant and dataset, and per step of producing an answer.

Latency is taken only from what actually ran. A response replayed from the cache returns at once, so
replayed answers (and replayed steps) are left out of the timing and the tables say how many were timed.
Cost counts every answer at what it cost when first run, with the newly spent part shown separately.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from bioscout.answer import CitedAnswer, Step
from bioscout.evals.metrics import MIN_FOR_P95, Record, format_cell, percentile

COST_COLUMNS = [
    ("variant", "Variant", "str"),
    ("dataset", "Dataset", "str"),
    ("n", "n", "str"),
    ("usd_per_query", "$/query", "usd"),
    ("usd_total", "$ total", "usd2"),
    ("usd_live", "$ newly spent", "usd2"),
    ("input_tokens_per_query", "Tokens in/query", "int"),
    ("output_tokens_per_query", "Tokens out/query", "int"),
    ("llm_calls_per_query", "LLM calls/query", "num"),
    ("tool_calls_per_query", "Tool calls/query", "num"),
    ("latency_p50_ms", "p50", "ms"),
    ("latency_p95_ms", "p95", "ms"),
    ("latency_n", "Answers timed", "str"),
]

STEP_COLUMNS = [
    ("variant", "Variant", "str"),
    ("step", "Step", "str"),
    ("runs_per_query", "Runs/query", "num"),
    ("llm_calls_per_query", "LLM calls/query", "num"),
    ("tool_calls_per_query", "Tool calls/query", "num"),
    ("usd_per_query", "$/query", "usd"),
    ("usd_share", "Share of cost", "pct"),
    ("latency_p50_ms", "p50", "ms"),
    ("latency_n", "Answers timed", "str"),
]


def _ran_live(step: Step, answer: CitedAnswer) -> bool:
    """Did this step do its work now? A step with LLM calls did if none was replayed; a step without
    any (a search) is taken to follow the rest of its answer."""
    if step.usage.llm_calls:
        return step.usage.cached_llm_calls == 0
    return answer.usage.cached_llm_calls == 0


def summarize_steps(records: list[Record]) -> list[dict[str, Any]]:
    """One row per (variant, step), over the answers that recorded their steps. A step that ran twice
    in an answer (a second search round, a re-check after a revision) is added up within that answer."""
    by_variant: dict[str, list[CitedAnswer]] = defaultdict(list)
    for r in records:
        if r.answer is not None and r.answer.steps:
            by_variant[r.variant].append(r.answer)

    rows: list[dict[str, Any]] = []
    for variant, answers in sorted(by_variant.items()):
        total_usd = sum(s.usage.usd or 0.0 for a in answers for s in a.steps)
        for name in dict.fromkeys(s.name for a in answers for s in a.steps):  # in the order steps first ran
            runs = [[s for s in a.steps if s.name == name] for a in answers]
            steps = [s for group in runs for s in group]
            usd = sum(s.usage.usd or 0.0 for s in steps)
            timed = [
                sum(s.usage.latency_ms for s in group)
                for group, a in zip(runs, answers, strict=True)
                if group and all(_ran_live(s, a) for s in group)
            ]
            rows.append(
                {
                    "variant": variant,
                    "step": name,
                    "runs_per_query": len(steps) / len(answers),
                    "llm_calls_per_query": sum(s.usage.llm_calls for s in steps) / len(answers),
                    "tool_calls_per_query": sum(s.usage.tool_calls for s in steps) / len(answers),
                    "usd_per_query": usd / len(answers),
                    "usd_share": usd / total_usd if total_usd else None,
                    "latency_p50_ms": percentile(timed, 0.5),
                    "latency_n": len(timed),
                }
            )
    return rows


def _table(columns: list[tuple[str, str, str]], rows: list[dict[str, Any]]) -> list[str]:
    head = "| " + " | ".join(label for _, label, _ in columns) + " |"
    rule = "|" + "|".join(" --- " for _ in columns) + "|"
    body = [
        "| " + " | ".join(format_cell(row.get(key), kind) for key, _, kind in columns) + " |" for row in rows
    ]
    return [head, rule, *body]


def render_costs(rows: list[dict[str, Any]], step_rows: list[dict[str, Any]]) -> str:
    """The cost sections of a run's report, to follow the metrics table."""
    lines = [
        "## Cost and latency",
        "",
        "Cost counts every answer at what it cost when first run; `$ newly spent` leaves out answers "
        "replayed from the cache. Latency is over answers that were not replayed (`Answers timed`), and "
        f"p95 is shown from {MIN_FOR_P95} timed answers up.",
        "",
        *_table(COST_COLUMNS, rows),
        "",
    ]
    if step_rows:
        lines += [
            "## Cost and latency by step",
            "",
            "Per answer, over the answers that recorded their steps. A step that ran twice in an answer is "
            "added up; its latency is over the answers in which it was not replayed.",
            "",
            *_table(STEP_COLUMNS, step_rows),
            "",
        ]
    return "\n".join(lines)
