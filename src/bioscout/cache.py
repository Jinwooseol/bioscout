"""SQLite response cache.

Every external API response is stored by a key derived from the request, so that
(1) public rate limits are respected, (2) evaluation runs are reproducible, and
(3) CI can replay recorded responses with no network (CacheMode.READONLY).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

# Credentials and identity params never change the response, so they never enter the key.
_IGNORED_PARAMS = {"api_key", "email", "tool"}


def make_key(url: str, params: dict[str, Any] | None) -> str:
    clean = {k: str(v) for k, v in sorted((params or {}).items()) if k not in _IGNORED_PARAMS}
    raw = json.dumps([url, clean], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class ResponseCache:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS responses ("
            " key TEXT PRIMARY KEY, url TEXT NOT NULL, body TEXT NOT NULL, stored_at REAL NOT NULL)"
        )
        self._db.commit()

    def get(self, key: str, max_age_s: float | None) -> str | None:
        row = self._db.execute("SELECT body, stored_at FROM responses WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        body, stored_at = row
        if max_age_s is not None and time.time() - stored_at > max_age_s:
            return None
        return body

    def put(self, key: str, url: str, body: str) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO responses (key, url, body, stored_at) VALUES (?, ?, ?, ?)",
            (key, url, body, time.time()),
        )
        self._db.commit()

    def __len__(self) -> int:
        return self._db.execute("SELECT COUNT(*) FROM responses").fetchone()[0]

    def close(self) -> None:
        self._db.close()
