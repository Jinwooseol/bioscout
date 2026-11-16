"""How the container starts: what it serves, and what it does with a data folder that root owns.
Nothing here needs a container: the calls that change ownership and identity are stand-ins."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

import pytest

from bioscout import container
from bioscout.config import Settings
from bioscout.container import Account

ROOT = Path(__file__).parent.parent
USER = Account(name="user", uid=1000, gid=1000, home="/home/user")


def test_it_serves_the_page_unless_told_to_serve_the_api_or_given_a_command():
    assert container.command([], {})[:3] == ["streamlit", "run", "app/streamlit_app.py"]
    assert container.command([], {"BIOSCOUT_SERVE": ""}) == list(container.PAGE)
    assert container.command([], {"BIOSCOUT_SERVE": " API "}) == ["bioscout-api"]
    # A command given to the container wins: docker run IMAGE bioscout-api, or a shell to look around.
    assert container.command(["bioscout-api"], {"BIOSCOUT_SERVE": "page"}) == ["bioscout-api"]
    assert container.command(["python", "-V"], {"BIOSCOUT_SERVE": "api"}) == ["python", "-V"]
    with pytest.raises(SystemExit, match="page or api"):
        container.command([], {"BIOSCOUT_SERVE": "both"})


def test_the_state_is_the_response_cache_and_wherever_the_count_of_questions_is(tmp_path: Path):
    together = Settings(cache_dir=tmp_path / "data")
    assert container.state_folders(together) == [tmp_path / "data"]
    apart = Settings(cache_dir=tmp_path / "data", quota_path=tmp_path / "count" / "quota.sqlite")
    assert container.state_folders(apart) == [tmp_path / "data", tmp_path / "count"]


def test_a_folder_is_handed_over_with_its_files_and_the_parents_made_for_it(tmp_path: Path):
    given: list[tuple[Path, int, int]] = []

    def chown(path: Path, uid: int, gid: int) -> None:
        given.append((path, uid, gid))

    # A mounted volume: it exists, holds a file from an earlier run and the file system's own folder.
    volume = tmp_path / "data"
    (volume / "lost+found").mkdir(parents=True)
    (volume / "quota.sqlite").write_bytes(b"")
    container.hand_over(volume, USER, chown)
    assert given == [(volume, 1000, 1000), (volume / "quota.sqlite", 1000, 1000)]

    # A folder that is not there yet: it is made, and so are its parents, which root would then own.
    given.clear()
    cache = tmp_path / "home" / ".cache" / "bioscout"
    container.hand_over(cache, USER, chown)
    assert cache.is_dir()
    assert [path for path, _, _ in given] == [cache, cache.parent, cache.parent.parent]


def test_as_root_the_folders_are_handed_over_before_root_is_given_up(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(os, "environ", dict(os.environ))  # the start-up sets HOME for the account
    os.environ["BIOSCOUT_CACHE_DIR"] = str(tmp_path / "data")
    os.environ["BIOSCOUT_QUOTA_PATH"] = str(tmp_path / "count" / "quota.sqlite")
    done: list[Any] = []
    account = Account(name="user", uid=1000, gid=1000, home=str(tmp_path / "home"))

    folders = container.become(
        account,
        chown=lambda path, uid, gid: done.append(path),
        give_up=lambda who: done.append(who),
    )
    assert folders == [tmp_path / "data", tmp_path / "count"] and all(f.is_dir() for f in folders)
    assert done == [tmp_path / "data", tmp_path / "count", account]  # root goes last, and for good
    assert os.environ["HOME"] == account.home and os.environ["USER"] == "user"


def test_the_start_up_runs_the_command_in_place_and_touches_nothing_unless_it_is_root(monkeypatch):
    ran: list[Any] = []
    monkeypatch.setattr(os, "execvp", lambda name, argv: ran.append((name, argv)))
    monkeypatch.setattr(container, "become", lambda account: ran.append(account))
    monkeypatch.setattr(container, "account_of", lambda: USER)
    monkeypatch.setenv("BIOSCOUT_SERVE", "api")

    monkeypatch.setattr("sys.argv", ["bioscout-serve"])
    monkeypatch.setattr(container, "is_root", lambda: False)  # Hugging Face: already uid 1000
    container.main()
    assert ran == [("bioscout-api", ["bioscout-api"])]

    ran.clear()
    monkeypatch.setattr("sys.argv", ["bioscout-serve", "python", "-V"])
    monkeypatch.setattr(container, "is_root", lambda: True)
    container.main()
    assert ran == [USER, ("python", ["python", "-V"])]


def test_the_fly_config_serves_the_api_from_one_machine_with_the_count_on_its_volume():
    """fly.toml cannot be deployed from here; this keeps it in step with what the code reads."""
    config = tomllib.loads((ROOT / "fly.toml").read_text(encoding="utf-8"))
    env, mount, service = config["env"], config["mounts"], config["http_service"]
    assert container.command([], env) == ["bioscout-api"]
    assert env["BIOSCOUT_CACHE_DIR"] == mount["destination"]  # the count is on the volume
    assert env["BIOSCOUT_API_CLIENT_HEADER"] == "Fly-Client-IP"
    assert env["BIOSCOUT_API_CORS_ORIGINS"].startswith("https://")
    assert service["internal_port"] == 7860 and service["force_https"]
    assert service["checks"][0]["path"] == "/health"
    # Nothing secret belongs in this file: keys are set with `fly secrets set`.
    assert not [name for name in env if "KEY" in name or "TOKEN" in name or "SECRET" in name]
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert 'ENTRYPOINT ["bioscout-serve"]' in dockerfile and "BIOSCOUT_API_PORT=7860" in dockerfile
