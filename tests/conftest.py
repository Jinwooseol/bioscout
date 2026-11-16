from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import respx

from bioscout.cache import ResponseCache
from bioscout.config import CacheMode, Settings
from bioscout.http import CachedHttp

FIXTURES = Path(__file__).parent / "fixtures"


def fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


@pytest.fixture(autouse=True)
def own_question_allowance(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    """No test counts against the real allowance of questions in the user's cache directory."""
    path = tmp_path_factory.mktemp("allowance") / "quota.sqlite"
    monkeypatch.setenv("BIOSCOUT_QUOTA_PATH", str(path))


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(ncbi_api_key="test-key", ncbi_email="dev@example.com", cache_dir=tmp_path)


@pytest.fixture
async def http(settings: Settings):
    client = CachedHttp(settings, cache=ResponseCache(settings.cache_dir / "t.sqlite"))
    yield client
    await client.aclose()


@pytest.fixture
def api():
    with respx.mock(assert_all_called=False) as mock:
        yield mock


def ok(body: str) -> httpx.Response:
    return httpx.Response(200, text=body)


__all__ = ["fixture", "ok", "CacheMode"]
