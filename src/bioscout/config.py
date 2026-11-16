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
    llm_model: str = "claude-opus-5-5"
    llm_effort: str = "medium"
    tools: str = "direct"  # "mcp": the agent reaches PubMed through the MCP server's tools
    # Tracing is off unless Langfuse keys or an OTLP endpoint are given (see tracing.py).
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = field(default=None, repr=False)
    langfuse_host: str = "https://cloud.langfuse.com"
    otlp_endpoint: str | None = None
    trace_content: bool = True  # False: no questions, prompts or replies in the spans
    # The cap on questions answered with the operator's API key (quota.py). It holds for every visitor
    # together and survives a restart; raise it here, never remove it.
    max_questions_per_day: int = 20
    max_questions_total: int = 200
    # Of the day's questions, how many one caller of the API may ask: it keeps one visitor from using
    # up the day for the others. It does not bound the bill; the two limits above do.
    max_questions_per_caller: int = 5
    quota_path: Path | None = None  # None: quota.sqlite in the cache directory
    # The HTTP API (api.py).
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    api_max_concurrent: int = 2  # questions answered at the same time
    api_timeout_s: float = 300.0  # one answer may take this long, then the request fails
    api_cors_origins: tuple[str, ...] = ()  # web pages on other origins allowed to call the API
    # The request header in which the host's proxy gives the caller's address (Fly-Client-IP on
    # Fly.io). None: the address the connection comes from, which behind a proxy is the proxy's.
    api_client_header: str | None = None
    examples_path: Path = Path("app/examples.json")  # the recorded answers the API serves

    def __post_init__(self) -> None:
        if self.tools not in ("direct", "mcp"):
            raise ValueError(f"BIOSCOUT_TOOLS must be direct or mcp, not {self.tools!r}")

    @property
    def through_mcp(self) -> bool:
        return self.tools == "mcp"

    @property
    def quota_file(self) -> Path:
        return self.quota_path or self.cache_dir / "quota.sqlite"

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
        quota_path = env.get("BIOSCOUT_QUOTA_PATH")
        origins = env.get("BIOSCOUT_API_CORS_ORIGINS", "")
        return cls(
            ncbi_api_key=env.get("NCBI_API_KEY") or None,
            ncbi_email=env.get("NCBI_EMAIL") or None,
            cache_dir=Path(cache_dir).expanduser() if cache_dir else cls().cache_dir,
            cache_mode=CacheMode(env.get("BIOSCOUT_CACHE_MODE", CacheMode.READWRITE)),
            cache_ttl_days=int(env.get("BIOSCOUT_CACHE_TTL_DAYS", 30)),
            llm_model=env.get("BIOSCOUT_MODEL") or cls.llm_model,
            llm_effort=env.get("BIOSCOUT_EFFORT") or cls.llm_effort,
            tools=env.get("BIOSCOUT_TOOLS") or cls.tools,
            langfuse_public_key=env.get("LANGFUSE_PUBLIC_KEY") or None,
            langfuse_secret_key=env.get("LANGFUSE_SECRET_KEY") or None,
            langfuse_host=env.get("LANGFUSE_HOST") or env.get("LANGFUSE_BASE_URL") or cls.langfuse_host,
            otlp_endpoint=env.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
            or env.get("OTEL_EXPORTER_OTLP_ENDPOINT")
            or None,
            trace_content=env.get("BIOSCOUT_TRACE_CONTENT", "on").lower() not in ("off", "0", "false", "no"),
            max_questions_per_day=int(env.get("BIOSCOUT_MAX_QUESTIONS_PER_DAY") or cls.max_questions_per_day),
            max_questions_total=int(env.get("BIOSCOUT_MAX_QUESTIONS_TOTAL") or cls.max_questions_total),
            max_questions_per_caller=int(
                env.get("BIOSCOUT_MAX_QUESTIONS_PER_CALLER") or cls.max_questions_per_caller
            ),
            quota_path=Path(quota_path).expanduser() if quota_path else None,
            api_host=env.get("BIOSCOUT_API_HOST") or cls.api_host,
            api_port=int(env.get("BIOSCOUT_API_PORT") or cls.api_port),
            api_max_concurrent=int(env.get("BIOSCOUT_API_MAX_CONCURRENT") or cls.api_max_concurrent),
            api_timeout_s=float(env.get("BIOSCOUT_API_TIMEOUT_S") or cls.api_timeout_s),
            api_cors_origins=tuple(o.strip().rstrip("/") for o in origins.split(",") if o.strip()),
            api_client_header=(env.get("BIOSCOUT_API_CLIENT_HEADER") or "").strip() or None,
            examples_path=Path(env.get("BIOSCOUT_EXAMPLES") or cls.examples_path),
        )
