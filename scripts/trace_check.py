"""Send one answer's trace, to check the tracing setup.

    uv run --extra trace python scripts/trace_check.py

Answers the first example question with tracing on and says where the trace went. The cache is read-only
unless BIOSCOUT_CACHE_MODE says otherwise, so a question that is already in the cache is replayed and
costs nothing; an uncached one fails instead of spending. Needs LANGFUSE_PUBLIC_KEY and
LANGFUSE_SECRET_KEY (or OTEL_EXPORTER_OTLP_ENDPOINT) in .env. A wrong key shows up as an export error
printed by the OpenTelemetry exporter.
"""

from __future__ import annotations

import asyncio
import os
import sys

os.environ.setdefault("BIOSCOUT_CACHE_MODE", "readonly")

from bioscout import demo, tracing  # noqa: E402
from bioscout.config import Settings  # noqa: E402


def main() -> int:
    target = tracing.configure(Settings.from_env())
    if target is None:
        print("Tracing is not configured: set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY in .env,")
        print("and install the extra with `uv sync --extra trace`.")
        return 1
    question = demo.EXAMPLE_QUESTIONS[0]
    result = asyncio.run(demo.ask(question))
    tracing.flush()
    steps = ", ".join(step.name for step in result.answer.steps)
    print(f"answered with {len(result.answer.sentences)} sentences ({steps})")
    print(f"trace sent to {target}: look for a trace named bioscout.answer, tagged demo")
    return 0


if __name__ == "__main__":
    sys.exit(main())
