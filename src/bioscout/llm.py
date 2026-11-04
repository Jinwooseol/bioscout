"""Claude calls that return JSON matching a schema, with refusal fallback, cost accounting and a cache.

Every request is keyed by its full content (model, effort, prompts, schema), so re-running an eval
replays recorded responses for free, and CI can replay them with no network (CacheMode.READONLY).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from anthropic import AsyncAnthropic

from bioscout.answer import Usage
from bioscout.cache import ResponseCache, make_key
from bioscout.config import CacheMode, Settings
from bioscout.http import CacheMissError

# USD per million tokens (input, output). Fallback models are listed so a rescued answer is still priced.
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}
FALLBACK_BETA = "server-side-fallback-2026-07-01"
_CACHE_URL = "anthropic:/v1/messages"


class LLMError(RuntimeError):
    """The model returned something unusable (truncated or invalid JSON)."""


def price_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    rates = PRICES_PER_MTOK.get(model)
    if rates is None:
        return None
    return (input_tokens * rates[0] + output_tokens * rates[1]) / 1_000_000


@dataclass
class LLMResult:
    data: dict[str, Any] | None  # parsed JSON, None when every model in the chain refused
    usage: Usage
    stop_reason: str
    refusal_category: str | None = None

    @property
    def refused(self) -> bool:
        return self.stop_reason == "refusal"


class LLM(Protocol):
    model: str
    effort: str

    async def json(
        self, *, system: str, user: str, schema: dict[str, Any], max_tokens: int = 16000
    ) -> LLMResult: ...


class ClaudeLLM:
    def __init__(
        self,
        settings: Settings,
        client: AsyncAnthropic | None = None,
        cache: ResponseCache | None = None,
    ) -> None:
        self.settings = settings
        self.model = settings.llm_model
        self.effort = settings.llm_effort
        self._client = client
        self.cache = cache if cache is not None else ResponseCache(settings.cache_dir / "llm.sqlite")

    @property
    def client(self) -> AsyncAnthropic:
        # Created lazily so replaying from the cache needs no API key.
        if self._client is None:
            self._client = AsyncAnthropic(max_retries=4)
        return self._client

    async def json(
        self, *, system: str, user: str, schema: dict[str, Any], max_tokens: int = 16000
    ) -> LLMResult:
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "output_config": {"effort": self.effort, "format": {"type": "json_schema", "schema": schema}},
        }
        # Haiku has no server-side fallback; elsewhere a refused request is retried on another model.
        if not self.model.startswith("claude-haiku"):
            request |= {"fallbacks": "default", "betas": [FALLBACK_BETA]}

        key = make_key(_CACHE_URL, {"request": json.dumps(request, sort_keys=True, ensure_ascii=False)})
        mode = self.settings.cache_mode
        if mode is not CacheMode.OFF:
            cached = self.cache.get(key, max_age_s=None)  # an LLM response never goes stale
            if cached is not None:
                result = _result_from_record(json.loads(cached))
                result.usage.cached_llm_calls = 1
                return result
            if mode is CacheMode.READONLY:
                raise CacheMissError(f"No recorded LLM response for model={self.model}")

        response = await self.client.beta.messages.create(**request)
        record = _record_from_response(response, self.model)
        if mode is not CacheMode.OFF:
            self.cache.put(key, _CACHE_URL, json.dumps(record, ensure_ascii=False))
        return _result_from_record(record)


def _record_from_response(response: Any, requested_model: str) -> dict[str, Any]:
    """Reduce an API response to what the harness needs; this is also what the cache stores."""
    if response.stop_reason == "max_tokens":
        raise LLMError("Response hit max_tokens before the JSON was complete")

    # With a fallback, usage.iterations lists every attempt; each is billed at its own model's rate.
    usage = response.usage
    attempts = [
        (getattr(it, "model", None) or requested_model, it.input_tokens, it.output_tokens, it.type)
        for it in (getattr(usage, "iterations", None) or [])
    ] or [(response.model, usage.input_tokens, usage.output_tokens, "message")]
    prices = [price_usd(m, i, o) for m, i, o, _ in attempts]

    record: dict[str, Any] = {
        "stop_reason": response.stop_reason,
        "refusal_category": None,
        "data": None,
        "model": response.model,
        "fell_back": any(kind == "fallback_message" for *_, kind in attempts),
        "input_tokens": sum(i for _, i, _, _ in attempts),
        "output_tokens": sum(o for _, _, o, _ in attempts),
        "usd": None if None in prices else sum(prices),
    }
    if response.stop_reason == "refusal":
        details = getattr(response, "stop_details", None)
        record["refusal_category"] = getattr(details, "category", None)
        return record

    text = next((b.text for b in response.content if b.type == "text"), "")
    try:
        record["data"] = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMError(f"Response was not valid JSON: {text[:200]!r}") from exc
    return record


def _result_from_record(record: dict[str, Any]) -> LLMResult:
    usage = Usage(
        usd=record["usd"],
        llm_calls=1,
        input_tokens=record["input_tokens"],
        output_tokens=record["output_tokens"],
        model=record["model"],
        fell_back=record["fell_back"],
    )
    return LLMResult(
        data=record["data"],
        usage=usage,
        stop_reason=record["stop_reason"],
        refusal_category=record["refusal_category"],
    )
