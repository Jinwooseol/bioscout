"""The cap on questions answered with the operator's key: per UTC day and in total, kept in a file."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from bioscout.config import Settings
from bioscout.quota import Quota, QuotaExceeded

NOON = datetime(2030, 1, 15, 12, 0, tzinfo=UTC)


class Clock:
    def __init__(self, now: datetime = NOON) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


def test_each_question_counts_until_the_day_is_used_up_and_the_next_day_starts_again(tmp_path: Path):
    clock = Clock()
    quota = Quota(tmp_path / "q.sqlite", per_day=2, total=10, now=clock)
    assert quota.status().left == 2
    assert quota.take().left_today == 1
    assert quota.take().left_today == 0

    with pytest.raises(QuotaExceeded) as refused:
        quota.take()
    assert refused.value.scope == "day"
    assert refused.value.retry_after_s == 12 * 3600  # from noon to the next UTC midnight
    assert refused.value.status.resets_at == datetime(2030, 1, 16, tzinfo=UTC)
    assert quota.status().used_total == 2  # a refused question is not counted

    clock.now += timedelta(days=1)
    after = quota.take()
    assert (after.used_today, after.used_total, after.left_today) == (1, 3, 1)


def test_the_total_holds_across_days_and_waiting_does_not_help(tmp_path: Path):
    clock = Clock()
    quota = Quota(tmp_path / "q.sqlite", per_day=2, total=3, now=clock)
    quota.take()
    quota.take()
    clock.now += timedelta(days=1)
    assert quota.take().left == 0  # one left today, none left in total

    with pytest.raises(QuotaExceeded) as refused:
        quota.take()
    assert refused.value.scope == "total" and refused.value.retry_after_s is None
    clock.now += timedelta(days=30)
    with pytest.raises(QuotaExceeded):
        quota.take()


def test_the_count_is_in_the_file_so_a_restart_or_another_process_sees_it(tmp_path: Path):
    path = tmp_path / "state" / "q.sqlite"  # the folder is made when missing
    Quota(path, per_day=1, total=5, now=Clock()).take()
    with pytest.raises(QuotaExceeded):
        Quota(path, per_day=1, total=5, now=Clock()).take()
    # The operator raising the limit lets the next question through; what was asked stays counted.
    assert Quota(path, per_day=3, total=5, now=Clock()).take().used_today == 2


def test_questions_arriving_together_cannot_pass_the_limit(tmp_path: Path):
    quota = Quota(tmp_path / "q.sqlite", per_day=5, total=100, now=Clock())

    def ask(_: int) -> bool:
        try:
            quota.take()
        except QuotaExceeded:
            return False
        return True

    with ThreadPoolExecutor(max_workers=16) as pool:
        let_in = list(pool.map(ask, range(40)))
    assert sum(let_in) == 5 and quota.status().used_today == 5


def test_a_limit_of_zero_closes_it_and_a_negative_one_is_a_mistake(tmp_path: Path):
    with pytest.raises(QuotaExceeded):
        Quota(tmp_path / "q.sqlite", per_day=0, total=10).take()
    with pytest.raises(ValueError):
        Quota(tmp_path / "q.sqlite", per_day=-1, total=10)


def test_the_limits_and_the_file_come_from_the_environment(tmp_path: Path, monkeypatch):
    for name in ("BIOSCOUT_MAX_QUESTIONS_PER_DAY", "BIOSCOUT_MAX_QUESTIONS_TOTAL", "BIOSCOUT_QUOTA_PATH"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("BIOSCOUT_CACHE_DIR", str(tmp_path))
    default = Settings.from_env(dotenv=None)
    assert (default.max_questions_per_day, default.max_questions_total) == (20, 200)
    assert default.quota_file == tmp_path / "quota.sqlite"  # beside the response cache

    monkeypatch.setenv("BIOSCOUT_MAX_QUESTIONS_PER_DAY", "0")  # zero is a limit, not "use the default"
    monkeypatch.setenv("BIOSCOUT_MAX_QUESTIONS_TOTAL", "7")
    monkeypatch.setenv("BIOSCOUT_QUOTA_PATH", str(tmp_path / "elsewhere.sqlite"))
    quota = Quota.from_settings(Settings.from_env(dotenv=None))
    assert (quota.per_day, quota.total, quota.path.name) == (0, 7, "elsewhere.sqlite")
