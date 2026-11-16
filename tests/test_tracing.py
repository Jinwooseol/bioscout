"""Tracing: the span tree of one answer, what the spans carry, and that it is off unless configured."""

from __future__ import annotations

import base64
import json
from typing import Any

import pytest

from bioscout import tracing
from bioscout.agent import ResearchAgent
from bioscout.config import Settings
from tests.fakes import FakeLLM, FakePubMed, reply
from tests.test_agent import plan, screen

TEXTS = {"10000001": "Drug X lowered mortality.", "10000002": "Drug X caused nausea."}
QUESTION = "Is X good?"


@pytest.fixture
def spans():
    """Trace into memory for one test; yields a function that returns the finished spans."""
    pytest.importorskip("opentelemetry.sdk", reason="the trace extra is not installed")
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    try:
        yield lambda content=True: (tracing.install(exporter, content=content, batch=False), exporter)[1]
    finally:
        tracing.reset()


async def answer_once() -> None:
    llm = FakeLLM(
        [plan("drug X mortality"), screen(["10000001", "10000002"]), reply("yes", ("X works.", ["10000001"]))]
    )
    agent = ResearchAgent(tracing.traced_llm(llm), tracing.traced_literature(FakePubMed(texts=TEXTS)))
    await agent.answer(QUESTION)


def tree(finished: list[Any]) -> list[tuple[str, str]]:
    """(span, its parent) pairs, sorted: spans that start within one clock tick have no reliable order."""
    names = {span.context.span_id: span.name for span in finished}
    pairs = [(span.name, names.get(span.parent.span_id) if span.parent else "") for span in finished]
    return sorted(pairs)


async def test_one_answer_is_one_trace_with_a_span_per_step_model_call_and_pubmed_call(spans):
    exporter = spans()
    with tracing.context(session_id="run-1", tags=["demo"]):
        await answer_once()
    finished = exporter.get_finished_spans()

    assert tree(finished) == sorted([
        ("bioscout.answer", ""),
        ("plan", "bioscout.answer"),
        ("claude.json", "plan"),
        ("search", "bioscout.answer"),
        ("pubmed.search", "search"),
        ("pubmed.search", "search"),
        ("pubmed.abstracts", "search"),
        ("screen", "bioscout.answer"),
        ("claude.json", "screen"),
        ("answer", "bioscout.answer"),
        ("claude.json", "answer"),
    ])  # fmt: skip
    assert len({span.context.trace_id for span in finished}) == 1
    by_name = {span.name: dict(span.attributes) for span in finished}

    root = by_name["bioscout.answer"]
    assert root["langfuse.observation.type"] == "agent" and root[tracing.INPUT] == QUESTION
    assert json.loads(root[tracing.OUTPUT])["sentences"] == [
        {"text": "X works.", "citations": ["10000001"], "verdict": None}
    ]
    assert by_name["search"]["langfuse.observation.type"] == "retriever"
    assert by_name["plan"]["langfuse.observation.metadata.llm_calls"] == "1"

    call = by_name["claude.json"]
    assert call["langfuse.observation.type"] == "generation"
    assert call["langfuse.observation.model.name"] == "claude-opus-5-5"
    assert json.loads(call["langfuse.observation.usage_details"]) == {
        "input": 100,
        "output": 50,
        "total": 150,
    }
    assert json.loads(call["langfuse.observation.cost_details"]) == {"total": 0.01}
    assert json.loads(call[tracing.INPUT])[1]["role"] == "user"

    assert sorted(json.loads(by_name["pubmed.abstracts"][tracing.OUTPUT])["found"]) == [
        "10000001",
        "10000002",
    ]
    # Langfuse wants the session and tags on every span of the trace, not only on its root.
    assert all(span.attributes["langfuse.session.id"] == "run-1" for span in finished)
    assert all(span.attributes["langfuse.trace.tags"] == ("demo",) for span in finished)


async def test_content_can_be_kept_out_of_the_spans(spans):
    exporter = spans(content=False)
    await answer_once()
    finished = exporter.get_finished_spans()
    assert finished and not [
        s.name for s in finished if tracing.INPUT in s.attributes or tracing.OUTPUT in s.attributes
    ]
    usage = [s.attributes["langfuse.observation.usage_details"] for s in finished if s.name == "claude.json"]
    assert len(usage) == 3  # tokens and cost are still sent


async def test_a_replayed_reply_costs_nothing_now_and_a_refusal_is_flagged(spans):
    exporter = spans()

    class Replaying(FakeLLM):
        async def json(self, **kwargs: Any):
            result = await super().json(**kwargs)
            result.usage.cached_llm_calls = 1
            return result

    await tracing.traced_llm(Replaying([{"ok": True}])).json(system="s", user="u", schema={})
    await tracing.traced_llm(FakeLLM([None])).json(system="s", user="u", schema={})
    replayed, refused = (dict(span.attributes) for span in exporter.get_finished_spans())

    assert json.loads(replayed["langfuse.observation.cost_details"]) == {"total": 0.0}
    assert replayed["langfuse.observation.metadata.original_usd"] == "0.01"
    assert refused["langfuse.observation.level"] == "WARNING"


async def test_an_error_inside_a_span_is_recorded_and_still_raised(spans):
    exporter = spans()
    with pytest.raises(RuntimeError, match="boom"), tracing.span("step"):
        raise RuntimeError("boom")
    (failed,) = exporter.get_finished_spans()
    assert failed.status.status_code.name == "ERROR"


async def test_tracing_is_off_and_wraps_nothing_unless_configured():
    assert not tracing.enabled() and tracing.configure(Settings()) is None
    llm, pubmed = FakeLLM([]), FakePubMed()
    assert tracing.traced_llm(llm) is llm and tracing.traced_literature(pubmed) is pubmed
    with tracing.span("anything", kind="agent", note=1) as current:
        current.input("question")
        current.output({"answer": 1})
    assert not tracing.enabled()


def test_langfuse_keys_or_an_otlp_endpoint_decide_where_spans_go():
    assert tracing.exporter_config(Settings()) is None
    assert tracing.exporter_config(Settings(langfuse_public_key="pk-lf-test")) is None  # both keys are needed
    assert tracing.exporter_config(Settings(otlp_endpoint="http://collector:4318")) == (None, {})

    settings = Settings(
        langfuse_public_key="pk-lf-test",
        langfuse_secret_key="sk-lf-test",
        langfuse_host="https://us.cloud.langfuse.com/",
    )
    endpoint, headers = tracing.exporter_config(settings)
    assert endpoint == "https://us.cloud.langfuse.com/api/public/otel/v1/traces"
    assert headers["x-langfuse-ingestion-version"] == "4"
    kind, token = headers["Authorization"].split()
    assert kind == "Basic" and base64.b64decode(token).decode() == "pk-lf-test:sk-lf-test"
    assert "sk-lf-test" not in repr(settings)  # the secret key stays out of logs
