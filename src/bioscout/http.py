"""Cached, rate-limited HTTP client shared by all data sources."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from typing import Any

import httpx

from bioscout import __version__
from bioscout.cache import ResponseCache, make_key
from bioscout.config import CacheMode, Settings


class CacheMissError(RuntimeError):
    """Raised in READONLY mode when a request has no recorded response."""


class UpstreamError(RuntimeError):
    """Raised when an external API keeps failing after retries."""


class RateLimiter:
    """Spaces requests to one host at least 1/rate seconds apart."""

    def __init__(self, per_second: float) -> None:
        self._interval = 1.0 / per_second
        self._lock = asyncio.Lock()
        self._next = 0.0

    async def wait(self) -> None:
        async with self._lock:
            now = time.monotonic()
            if self._next > now:
                await asyncio.sleep(self._next - now)
            self._next = max(now, self._next) + self._interval


class CachedHttp:
    _RETRY_STATUS = {429, 500, 502, 503, 504}

    def __init__(
        self,
        settings: Settings,
        cache: ResponseCache | None = None,
        client: httpx.AsyncClient | None = None,
        rate_limits: Mapping[str, float] | None = None,
    ) -> None:
        self.settings = settings
        self.cache = cache if cache is not None else ResponseCache(settings.cache_dir / "http.sqlite")
        self.client = client or httpx.AsyncClient(
            timeout=settings.http_timeout_s,
            headers={"User-Agent": f"bioscout/{__version__} (+https://github.com/Jinwooseol/bioscout)"},
            follow_redirects=True,
        )
        self._limiters = {host: RateLimiter(rps) for host, rps in (rate_limits or {}).items()}
        self.stats = {"hits": 0, "misses": 0}

    async def get_text(self, url: str, params: dict[str, Any] | None = None) -> str:
        key = make_key(url, params)
        mode = self.settings.cache_mode
        if mode is not CacheMode.OFF:
            max_age = None if mode is CacheMode.READONLY else self.settings.cache_ttl_days * 86400
            cached = self.cache.get(key, max_age)
            if cached is not None:
                self.stats["hits"] += 1
                return cached
            if mode is CacheMode.READONLY:
                raise CacheMissError(f"No recorded response for {url} {params}")
        self.stats["misses"] += 1
        body = await self._fetch(url, params)
        if mode is not CacheMode.OFF:
            self.cache.put(key, url, body)
        return body

    async def _fetch(self, url: str, params: dict[str, Any] | None, attempts: int = 4) -> str:
        host = httpx.URL(url).host
        limiter = self._limiters.get(host)
        last_error: Exception | None = None
        for attempt in range(attempts):
            if limiter:
                await limiter.wait()
            try:
                resp = await self.client.get(url, params=params)
            except httpx.TransportError as exc:
                last_error = exc
            else:
                if resp.status_code == 200:
                    return resp.text
                if resp.status_code not in self._RETRY_STATUS:
                    resp.raise_for_status()
                last_error = httpx.HTTPStatusError(
                    f"HTTP {resp.status_code}", request=resp.request, response=resp
                )
            await asyncio.sleep(0.5 * 2**attempt)
        raise UpstreamError(f"{url} failed after {attempts} attempts: {last_error}")

    async def aclose(self) -> None:
        await self.client.aclose()
