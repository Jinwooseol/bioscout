"""How the container starts. This is the image's entrypoint, `bioscout-serve`.

    bioscout-serve                 the demo page, or the API when BIOSCOUT_SERVE=api
    bioscout-serve COMMAND ...     that command instead (docker run IMAGE bioscout-api)

Hosts differ in two ways the image should not have to know about. Some cannot change the command an
image runs, so what to serve is also read from the environment. And some mount a volume that root
owns, where the user the server runs as (uid 1000) could not write the count of questions. So when
the container starts as root, the folders that hold state are made and handed to that user, and root
is given up before anything is served. Started as another user (Hugging Face starts the container as
uid 1000), nothing is changed.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from bioscout.config import Settings

PAGE = (
    "streamlit",
    "run",
    "app/streamlit_app.py",
    "--server.port=7860",
    "--server.address=0.0.0.0",
    "--server.headless=true",
    "--browser.gatherUsageStats=false",
)
API = ("bioscout-api",)
USER = "user"  # the account the Dockerfile creates with uid 1000


@dataclass(frozen=True)
class Account:
    name: str
    uid: int
    gid: int
    home: str


def command(args: Sequence[str], env: Mapping[str, str]) -> list[str]:
    """What to run: the command given, or else what BIOSCOUT_SERVE names (the page when unset)."""
    if args:
        return list(args)
    serve = env.get("BIOSCOUT_SERVE", "").strip().lower() or "page"
    if serve == "page":
        return list(PAGE)
    if serve == "api":
        return list(API)
    raise SystemExit(f"BIOSCOUT_SERVE must be page or api, not {serve!r}")


def state_folders(settings: Settings) -> list[Path]:
    """The folders the server writes to: the response cache, and where the count of questions is."""
    folders = [settings.cache_dir]
    if settings.quota_file.parent not in folders:
        folders.append(settings.quota_file.parent)
    return folders


def hand_over(folder: Path, account: Account, chown: Callable[[Path, int, int], None] | None = None) -> None:
    """Make the folder and give it to the account, with the files in it and any parent made on the way.
    Folders inside it are left alone (a volume's lost+found)."""
    chown = chown or os.chown  # looked up here: os.chown does not exist on Windows, where tests run
    made = [path for path in (folder, *folder.parents) if not path.exists()]
    folder.mkdir(parents=True, exist_ok=True)
    files = sorted(path for path in folder.iterdir() if path.is_file())
    for path in dict.fromkeys((folder, *made, *files)):
        chown(path, account.uid, account.gid)


def account_of(name: str = USER) -> Account:
    import pwd  # POSIX only: this runs in the container

    entry = pwd.getpwnam(name)
    return Account(entry.pw_name, entry.pw_uid, entry.pw_gid, entry.pw_dir)


def give_up_root(account: Account) -> None:
    os.setgroups([account.gid])
    os.setgid(account.gid)
    os.setuid(account.uid)


def become(
    account: Account,
    chown: Callable[[Path, int, int], None] | None = None,
    give_up: Callable[[Account], None] = give_up_root,
) -> list[Path]:
    """As root: hand the state folders to the account, then stop being root. Returns the folders."""
    # Before the settings are read: the default cache folder is in the home directory.
    os.environ.update(HOME=account.home, USER=account.name, LOGNAME=account.name)
    folders = state_folders(Settings.from_env(dotenv=None))
    for folder in folders:
        hand_over(folder, account, chown)
    give_up(account)
    return folders


def is_root() -> bool:
    return hasattr(os, "geteuid") and os.geteuid() == 0


def main() -> None:
    argv = command(sys.argv[1:], os.environ)
    if is_root():
        become(account_of())
    try:
        os.execvp(argv[0], argv)
    except FileNotFoundError:
        raise SystemExit(f"bioscout-serve: command not found: {argv[0]}") from None


if __name__ == "__main__":
    main()
