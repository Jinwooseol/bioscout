"""Publish the demo page as a Hugging Face Docker Space.

    uv run --with huggingface_hub python scripts/deploy_space.py OWNER/NAME

Uploads only what the image needs (the Dockerfile, the lock file, src/ and app/) plus the Space's own
README (space/README.md), as one commit, then waits until the Space is running. It uses the Hugging Face
login saved on this machine (`hf auth login`); no token is read from or written to the repository.

No Claude API key is set on the Space, so the public page shows the recorded example answers only and
cannot spend money. `--dry-run` lists the files without uploading anything.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
FILES = ("Dockerfile", ".dockerignore", "pyproject.toml", "uv.lock", "LICENSE")
FOLDERS = ("src", "app")
SKIP = ("__pycache__", "*.pyc", ".env*", "*.sqlite", ".cache")
FAILED = ("BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR", "NO_APP_FILE", "PAUSED")


def stage(dest: Path, root: Path = ROOT) -> list[str]:
    """Copy what the Space needs into dest and return the staged paths."""
    for name in FILES:
        shutil.copy2(root / name, dest / name)
    for name in FOLDERS:
        shutil.copytree(root / name, dest / name, ignore=shutil.ignore_patterns(*SKIP))
    shutil.copy2(root / "space" / "README.md", dest / "README.md")
    return sorted(path.relative_to(dest).as_posix() for path in dest.rglob("*") if path.is_file())


def revision(root: Path = ROOT) -> str:
    """The commit being deployed, marked when the working tree has changes that are not committed."""

    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout

    return git("rev-parse", "--short", "HEAD").strip() + ("+changes" if git("status", "--porcelain") else "")


def page_url(repo_id: str) -> str:
    return "https://" + repo_id.lower().replace("/", "-").replace("_", "-").replace(".", "-") + ".hf.space"


def wait_until_running(api, repo_id: str, timeout_s: int = 1200) -> bool:
    """Follow the Space's build until it runs, fails or the time is up."""
    seen = None
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        stage_now = str(api.get_space_runtime(repo_id).stage)
        if stage_now != seen:
            print(f"  {stage_now}", flush=True)
            seen = stage_now
        if stage_now == "RUNNING":
            return True
        if stage_now in FAILED:
            return False
        time.sleep(10)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish the demo page as a Hugging Face Docker Space.")
    parser.add_argument("repo_id", help="the Space, as owner/name")
    parser.add_argument("--dry-run", action="store_true", help="list the files and upload nothing")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        files = stage(Path(tmp))
        print(f"{len(files)} files for the Space:")
        for name in files:
            print(f"  {name}")
        if args.dry_run:
            return 0

        from huggingface_hub import HfApi  # imported here, so a dry run needs no extra package

        api = HfApi()
        api.create_repo(args.repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
        api.upload_folder(
            folder_path=tmp,
            repo_id=args.repo_id,
            repo_type="space",
            commit_message=f"Deploy bioscout {revision()}",
            delete_patterns="*",  # files no longer in the upload are removed from the Space
        )

    print(f"uploaded to https://huggingface.co/spaces/{args.repo_id}, waiting for the build:")
    if not wait_until_running(api, args.repo_id):
        print("the Space did not reach the running state; see its build log")
        return 1
    print(f"running at {page_url(args.repo_id)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
