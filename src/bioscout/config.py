"""Runtime settings, read from environment variables (and an optional .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path


class CacheMode(StrEnum):
    READWRITE = "readwrite"  # default: serve from cache, fetch and store on miss
    READONLY = "readonly"  # replay only: a cache miss is an error (CI, reproducible evals)
    OFF = "off"  # always hit the network, store nothing


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader so the MCP server works without python-dotenv."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class Settings:
    ncbi_api_key: str | None = None
    ncbi_email: str | None = None
    ncbi_tool: str = "bioscout"
    cache_dir: Path = field(default_factory=lambda: Path.home() / ".cache" / "bioscout")
    cache_mode: CacheMode = CacheMode.READWRITE
    cache_ttl_days: int = 30
    http_timeout_s: float = 20.0

    @property
    def ncbi_requests_per_second(self) -> float:
        # NCBI policy: 3 req/s without an API key, 10 req/s with one.
        return 9.0 if self.ncbi_api_key else 2.5

    @classmethod
    def from_env(cls, dotenv: Path | None = Path(".env")) -> Settings:
        if dotenv is not None:
            _load_dotenv(dotenv)
        env = os.environ
        cache_dir = env.get("BIOSCOUT_CACHE_DIR")
        return cls(
            ncbi_api_key=env.get("NCBI_API_KEY") or None,
            ncbi_email=env.get("NCBI_EMAIL") or None,
            cache_dir=Path(cache_dir).expanduser() if cache_dir else cls().cache_dir,
            cache_mode=CacheMode(env.get("BIOSCOUT_CACHE_MODE", CacheMode.READWRITE)),
            cache_ttl_days=int(env.get("BIOSCOUT_CACHE_TTL_DAYS", 30)),
        )
