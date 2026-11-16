"""A cap on the questions answered with the operator's Claude API key: so many per UTC day, and so
many in all.

The count lives in a SQLite file, so a restart does not reset it, and every process that opens the same
file (the API, the demo page) draws on the same allowance. The demo page's per-session limit resets when
the visitor refreshes; this one does not, so it is what bounds the bill of a public deployment.

A question counts once it is let through, whether or not its answer succeeds: a failed answer may
already have paid for some of its steps, and giving the question back would let failures be retried
without end.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

from bioscout.config import Settings

Scope = Literal["day", "total"]


def utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class QuotaStatus:
    per_day: int
    total: int
    used_today: int
    used_total: int
    resets_at: datetime  # when the daily count starts again (the next UTC midnight)

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
    """The allowance is used up. `scope` says which one: today's, or the total."""

    def __init__(self, scope: Scope, status: QuotaStatus, now: datetime) -> None:
        super().__init__(f"question allowance used up ({scope})")
        self.scope: Scope = scope
        self.status = status
        # Waiting helps only for the daily allowance; the total is raised by the operator or not at all.
        wait = (status.resets_at - now).total_seconds()
        self.retry_after_s: int | None = max(1, round(wait)) if scope == "day" else None


class Quota:
    def __init__(self, path: Path, per_day: int, total: int, now: Callable[[], datetime] = utc_now) -> None:
        if per_day < 0 or total < 0:
            raise ValueError("the question limits cannot be negative")
        self.path = path
        self.per_day = per_day
        self.total = total
        self._now = now
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db:
            db.execute("CREATE TABLE IF NOT EXISTS questions (day TEXT PRIMARY KEY, asked INTEGER NOT NULL)")

    @classmethod
    def from_settings(cls, settings: Settings) -> Quota:
        return cls(settings.quota_file, settings.max_questions_per_day, settings.max_questions_total)

    def _connect(self) -> sqlite3.Connection:
        # One short-lived connection per call: safe from any thread and any process on the same file.
        return sqlite3.connect(self.path, timeout=10, isolation_level=None)

    def _read(self, db: sqlite3.Connection, now: datetime) -> QuotaStatus:
        day = now.date().isoformat()
        today = db.execute("SELECT asked FROM questions WHERE day = ?", (day,)).fetchone()
        (used_total,) = db.execute("SELECT COALESCE(SUM(asked), 0) FROM questions").fetchone()
        midnight = datetime.combine(now.date() + timedelta(days=1), datetime.min.time(), tzinfo=UTC)
        return QuotaStatus(self.per_day, self.total, today[0] if today else 0, used_total, midnight)

    def status(self) -> QuotaStatus:
        with closing(self._connect()) as db:
            return self._read(db, self._now())

    def take(self) -> QuotaStatus:
        """Count one question, or raise QuotaExceeded and count nothing. The check and the count are one
        transaction, so questions arriving together cannot pass the limit between them."""
        now = self._now()
        with closing(self._connect()) as db:
            db.execute("BEGIN IMMEDIATE")  # takes the write lock before reading
            try:
                before = self._read(db, now)
                if before.left_total <= 0:
                    raise QuotaExceeded("total", before, now)
                if before.left_today <= 0:
                    raise QuotaExceeded("day", before, now)
                db.execute(
                    "INSERT INTO questions (day, asked) VALUES (?, 1) "
                    "ON CONFLICT(day) DO UPDATE SET asked = asked + 1",
                    (now.date().isoformat(),),
                )
                after = self._read(db, now)
            except BaseException:
                db.execute("ROLLBACK")
                raise
            db.execute("COMMIT")
            return after
