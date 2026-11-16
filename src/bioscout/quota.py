"""A cap on the questions answered with the operator's Claude API key: so many per UTC day, and so
many in all. Within the day, one caller may ask only so many.

The count lives in a SQLite file, so a restart does not reset it, and every process that opens the same
file (the API, the demo page) draws on the same allowance. The demo page's per-session limit resets when
the visitor refreshes; this one does not, so it is what bounds the bill of a public deployment.

A question counts once it is let through, whether or not its answer succeeds: a failed answer may
already have paid for some of its steps, and giving the question back would let failures be retried
without end.

The limit per caller keeps one visitor from using up the day for everyone else. It does not bound the
bill, since a caller who changes address gets a new count; the allowance per day and in total does. A
caller is kept as a salted hash and only for the day, so the file never holds an address.
"""

from __future__ import annotations

import hashlib
import secrets
import sqlite3
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

from bioscout.config import Settings

Scope = Literal["day", "total", "caller"]


def utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class QuotaStatus:
    per_day: int
    total: int
    used_today: int
    used_total: int
    resets_at: datetime  # when the daily counts start again (the next UTC midnight)
    per_caller: int  # how many of the day's questions one caller may ask

    @property
    def left_today(self) -> int:
        return max(0, self.per_day - self.used_today)

    @property
    def left_total(self) -> int:
        return max(0, self.total - self.used_total)

    @property
    def left(self) -> int:
        """Questions that can still be asked now."""
        return min(self.left_today, self.left_total)


class QuotaExceeded(Exception):
    """An allowance is used up. `scope` says which: today's, the total, or this caller's for today."""

    def __init__(self, scope: Scope, status: QuotaStatus, now: datetime) -> None:
        super().__init__(f"question allowance used up ({scope})")
        self.scope: Scope = scope
        self.status = status
        # Waiting helps for the daily counts; the total is raised by the operator or not at all.
        wait = (status.resets_at - now).total_seconds()
        self.retry_after_s: int | None = None if scope == "total" else max(1, round(wait))


class Quota:
    def __init__(
        self,
        path: Path,
        per_day: int,
        total: int,
        per_caller: int | None = None,
        now: Callable[[], datetime] = utc_now,
    ) -> None:
        """`per_caller` left out means one caller may ask all of the day's questions."""
        per_caller = per_day if per_caller is None else per_caller
        if per_day < 0 or total < 0 or per_caller < 0:
            raise ValueError("the question limits cannot be negative")
        self.path = path
        self.per_day = per_day
        self.total = total
        self.per_caller = per_caller
        self._now = now
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db:
            db.execute("CREATE TABLE IF NOT EXISTS questions (day TEXT PRIMARY KEY, asked INTEGER NOT NULL)")
            db.execute(
                "CREATE TABLE IF NOT EXISTS callers "
                "(day TEXT NOT NULL, caller TEXT NOT NULL, asked INTEGER NOT NULL, PRIMARY KEY (day, caller))"
            )
            # The salt of the caller hashes: made once per file, so a hash means nothing outside it.
            db.execute("CREATE TABLE IF NOT EXISTS salt (id INTEGER PRIMARY KEY CHECK (id = 1), value TEXT)")
            db.execute("INSERT OR IGNORE INTO salt (id, value) VALUES (1, ?)", (secrets.token_hex(16),))

    @classmethod
    def from_settings(cls, settings: Settings) -> Quota:
        return cls(
            settings.quota_file,
            settings.max_questions_per_day,
            settings.max_questions_total,
            settings.max_questions_per_caller,
        )

    def _connect(self) -> sqlite3.Connection:
        # One short-lived connection per call: safe from any thread and any process on the same file.
        return sqlite3.connect(self.path, timeout=10, isolation_level=None)

    def _read(self, db: sqlite3.Connection, now: datetime) -> QuotaStatus:
        day = now.date().isoformat()
        today = db.execute("SELECT asked FROM questions WHERE day = ?", (day,)).fetchone()
        (used_total,) = db.execute("SELECT COALESCE(SUM(asked), 0) FROM questions").fetchone()
        midnight = datetime.combine(now.date() + timedelta(days=1), datetime.min.time(), tzinfo=UTC)
        used_today = today[0] if today else 0
        return QuotaStatus(self.per_day, self.total, used_today, used_total, midnight, self.per_caller)

    def _key(self, db: sqlite3.Connection, caller: str) -> str:
        (salt,) = db.execute("SELECT value FROM salt WHERE id = 1").fetchone()
        return hashlib.sha256(f"{salt}:{caller}".encode()).hexdigest()[:32]

    def status(self) -> QuotaStatus:
        with closing(self._connect()) as db:
            return self._read(db, self._now())

    def take(self, caller: str | None = None) -> QuotaStatus:
        """Count one question, or raise QuotaExceeded and count nothing. The check and the count are one
        transaction, so questions arriving together cannot pass a limit between them.

        `caller` is whatever tells callers apart (the API gives the network address). With it, the
        question also counts against that caller's share of the day; without it (the demo page, which
        cannot tell its visitors apart) only the allowance for everyone applies."""
        now = self._now()
        day = now.date().isoformat()
        with closing(self._connect()) as db:
            db.execute("BEGIN IMMEDIATE")  # takes the write lock before reading
            try:
                before = self._read(db, now)
                if before.left_total <= 0:
                    raise QuotaExceeded("total", before, now)
                if before.left_today <= 0:
                    raise QuotaExceeded("day", before, now)
                if caller is not None:
                    key = self._key(db, caller)
                    asked = db.execute(
                        "SELECT asked FROM callers WHERE day = ? AND caller = ?", (day, key)
                    ).fetchone()
                    if (asked[0] if asked else 0) >= self.per_caller:
                        raise QuotaExceeded("caller", before, now)
                    db.execute("DELETE FROM callers WHERE day < ?", (day,))  # earlier days are of no use
                    db.execute(
                        "INSERT INTO callers (day, caller, asked) VALUES (?, ?, 1) "
                        "ON CONFLICT(day, caller) DO UPDATE SET asked = asked + 1",
                        (day, key),
                    )
                db.execute(
                    "INSERT INTO questions (day, asked) VALUES (?, 1) "
                    "ON CONFLICT(day) DO UPDATE SET asked = asked + 1",
                    (day,),
                )
                after = self._read(db, now)
            except BaseException:
                db.execute("ROLLBACK")
                raise
            db.execute("COMMIT")
            return after
