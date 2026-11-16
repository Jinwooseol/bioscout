"""Tracing: what one answer did, step by step, sent as OpenTelemetry spans.

Off unless configured. With LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY set (and LANGFUSE_HOST for a
region other than the EU cloud), spans go to Langfuse's OTLP endpoint; with OTEL_EXPORTER_OTLP_ENDPOINT
set they go to any OTLP backend. The OpenTelemetry packages are an optional extra (`trace`). Without
configuration every function here does nothing and wraps nothing, so the rest of the code calls them
unconditionally.

One answer is one trace:

    bioscout.answer            agent        the question in, the cited answer out
      plan, screen, answer     chain        one span per step of the graph, with what the step cost
        claude.json            generation   model, tokens, cost, the prompt and the reply
      search                   retriever
        pubmed.search          tool         the query in, the PMIDs out
      verify                   guardrail

Attribute names follow Langfuse's OpenTelemetry mapping (langfuse.observation.*, langfuse.trace.*).
BIOSCOUT_TRACE_CONTENT=off keeps questions, prompts and replies out of the spans: names, counts, tokens
and cost are still sent.
"""

from __future__ import annotations

import base64
import json
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from bioscout.config import Settings
from bioscout.llm import LLM, LLMResult
from bioscout.models import AbstractBatch, PmidCheck, SearchResult
from bioscout.sources.pubmed import Literature

INPUT = "langfuse.observation.input"
OUTPUT = "langfuse.observation.output"
STEP_KINDS = {"search": "retriever", "verify": "guardrail"}  # the other steps are "chain"


@dataclass
class _State:
    tracer: Any
    provider: Any
    content: bool  # send questions, prompts and replies, not only names and numbers
    target: str  # where spans go, for a log line


_state: _State | None = None
# Session, tags and trace metadata. Langfuse wants them on every span of a trace, not only on its root.
_trace_attributes: ContextVar[dict[str, Any]] = ContextVar("bioscout_trace_attributes", default={})  # noqa: B039


def exporter_config(settings: Settings) -> tuple[str | None, dict[str, str]] | None:
    """Where spans go. (endpoint, headers) for Langfuse; (None, {}) for a backend named by the standard
    OTEL_ variables, which the exporter reads itself; None when tracing is not configured."""
    public, secret = settings.langfuse_public_key, settings.langfuse_secret_key
    if public and secret:
        token = base64.b64encode(f"{public}:{secret}".encode()).decode()
        endpoint = f"{settings.langfuse_host.rstrip('/')}/api/public/otel/v1/traces"
        # Without the ingestion version Langfuse may show directly sent spans several minutes late.
        return endpoint, {"Authorization": f"Basic {token}", "x-langfuse-ingestion-version": "4"}
    if settings.otlp_endpoint:
        return None, {}
    return None


def install(exporter: Any, content: bool = True, batch: bool = True, target: str = "an exporter") -> None:
    """Start tracing to this exporter. `configure` calls it; tests pass an in-memory exporter."""
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor

    global _state
    # The provider is kept here rather than set as the process-wide one, so nothing else is traced.
    provider = TracerProvider(resource=Resource.create({"service.name": "bioscout"}))
    provider.add_span_processor((BatchSpanProcessor if batch else SimpleSpanProcessor)(exporter))
    _state = _State(provider.get_tracer("bioscout"), provider, content, target)


def configure(settings: Settings) -> str | None:
    """Turn tracing on when it is configured. Returns where spans go, or None when tracing stays off."""
    if _state is not None:
        return _state.target
    config = exporter_config(settings)
    if config is None:
        return None
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    except ImportError:
        warnings.warn("Tracing is configured but not installed: run `uv sync --extra trace`.", stacklevel=2)
        return None
    endpoint, headers = config
    target = f"Langfuse at {settings.langfuse_host}" if endpoint else f"OTLP at {settings.otlp_endpoint}"
    install(
        OTLPSpanExporter(endpoint=endpoint, headers=headers or None), settings.trace_content, target=target
    )
    return target


def enabled() -> bool:
    return _state is not None


def flush() -> None:
    """Send what is buffered now. The provider also flushes when the process exits."""
    if _state is not None:
        _state.provider.force_flush()


def reset() -> None:
    """Stop tracing (for tests)."""
    global _state
    if _state is not None:
        _state.provider.shutdown()
    _state = None


def _as_json(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)


class _Span:
    def __init__(self, span: Any, content: bool) -> None:
        self._span = span
        self._content = content

    def set(self, **metadata: Any) -> None:
        """Metadata shown beside the observation. Langfuse reads these as strings."""
        for key, value in metadata.items():
            if value is not None:
                self._span.set_attribute(f"langfuse.observation.metadata.{key}", str(value))

    def input(self, value: Any) -> None:
        if self._content:
            self._span.set_attribute(INPUT, _as_json(value))

    def output(self, value: Any) -> None:
        if self._content:
            self._span.set_attribute(OUTPUT, _as_json(value))

    def generation(self, result: LLMResult, parameters: dict[str, Any]) -> None:
        """What a model call used and returned."""
        usage = result.usage
        replayed = bool(usage.cached_llm_calls)
        tokens = {
            "input": usage.input_tokens,
            "output": usage.output_tokens,
            "total": usage.input_tokens + usage.output_tokens,
        }
        if usage.model:
            self._span.set_attribute("langfuse.observation.model.name", usage.model)
        self._span.set_attribute("langfuse.observation.model.parameters", json.dumps(parameters))
        self._span.set_attribute("langfuse.observation.usage_details", json.dumps(tokens))
        if usage.usd is not None:
            # A reply replayed from the cache costs nothing now; its first cost is kept beside it.
            spent = 0.0 if replayed else usage.usd
            self._span.set_attribute("langfuse.observation.cost_details", json.dumps({"total": spent}))
        self.set(
            replayed=replayed,
            original_usd=usage.usd if replayed else None,
            fell_back=usage.fell_back or None,
            stop_reason=result.stop_reason,
        )
        if result.refused:
            self._span.set_attribute("langfuse.observation.level", "WARNING")
            self._span.set_attribute("langfuse.observation.status_message", "the model refused")
        self.output(result.data)


class _NoSpan:
    """What `span` yields when tracing is off."""

    def set(self, **metadata: Any) -> None: ...
    def input(self, value: Any) -> None: ...
    def output(self, value: Any) -> None: ...
    def generation(self, result: LLMResult, parameters: dict[str, Any]) -> None: ...


_NO_SPAN = _NoSpan()


@contextmanager
def span(name: str, kind: str = "span", **metadata: Any) -> Iterator[_Span | _NoSpan]:
    """A span under the current one. `kind` is Langfuse's observation type (agent, chain, generation, ...).
    An exception leaving the block marks the span as failed and goes on up."""
    state = _state
    if state is None:
        yield _NO_SPAN
        return
    with state.tracer.start_as_current_span(name) as current:
        current.set_attribute("langfuse.observation.type", kind)
        for key, value in _trace_attributes.get().items():
            current.set_attribute(key, value)
        wrapped = _Span(current, state.content)
        wrapped.set(**metadata)
        yield wrapped


@contextmanager
def context(session_id: str | None = None, tags: list[str] | None = None, **metadata: Any) -> Iterator[None]:
    """Session, tags and trace metadata for every span started inside the block (and tasks started there)."""
    attributes = dict(_trace_attributes.get())
    if session_id:
        attributes["langfuse.session.id"] = session_id
    if tags:
        attributes["langfuse.trace.tags"] = [*attributes.get("langfuse.trace.tags", []), *tags]
    for key, value in metadata.items():
        if value is not None:
            attributes[f"langfuse.trace.metadata.{key}"] = str(value)
    token = _trace_attributes.set(attributes)
    try:
        yield
    finally:
        _trace_attributes.reset(token)


class TracedLLM:
    """An LLM whose every call is a generation span."""

    def __init__(self, inner: LLM) -> None:
        self.inner = inner
        self.model = inner.model
        self.effort = inner.effort

    async def json(
        self, *, system: str, user: str, schema: dict[str, Any], max_tokens: int = 16000
    ) -> LLMResult:
        with span("claude.json", kind="generation") as current:
            current.input([{"role": "system", "content": system}, {"role": "user", "content": user}])
            result = await self.inner.json(system=system, user=user, schema=schema, max_tokens=max_tokens)
            current.generation(result, {"effort": self.effort, "max_tokens": max_tokens})
            return result


class TracedLiterature:
    """PubMed calls as tool spans. The abstracts themselves are not copied into the span: the model call
    that reads them carries them in its prompt."""

    def __init__(self, inner: Literature) -> None:
        self.inner = inner

    async def search(self, query: str, max_results: int = 10) -> SearchResult:
        with span("pubmed.search", kind="tool") as current:
            current.input({"query": query, "max_results": max_results})
            result = await self.inner.search(query, max_results=max_results)
            current.output({"total_found": result.total_found, "pmids": [a.pmid for a in result.articles]})
            current.set(results=len(result.articles), total_found=result.total_found)
            return result

    async def abstracts(self, pmids: list[str]) -> AbstractBatch:
        with span("pubmed.abstracts", kind="tool") as current:
            current.input({"pmids": pmids})
            batch = await self.inner.abstracts(pmids)
            current.output({"found": [a.pmid for a in batch.articles], "not_found": batch.not_found})
            current.set(requested=len(pmids), found=len(batch.articles))
            return batch

    async def check_pmids(self, pmids: list[str]) -> PmidCheck:
        with span("pubmed.check_pmids", kind="tool") as current:
            current.input({"pmids": pmids})
            check = await self.inner.check_pmids(pmids)
            current.output({"missing": check.missing})
            current.set(checked=len(pmids), missing=len(check.missing))
            return check


def traced_llm(llm: LLM) -> LLM:
    """The LLM itself while tracing is off, so nothing changes; a traced wrapper once it is on."""
    return TracedLLM(llm) if _state is not None else llm


def traced_literature(pubmed: Literature) -> Literature:
    return TracedLiterature(pubmed) if _state is not None else pubmed
