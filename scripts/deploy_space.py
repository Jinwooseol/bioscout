"""Publish the demo as a Hugging Face Space.

    uv run --with huggingface_hub python scripts/deploy_space.py OWNER/NAME [--sdk static|docker]

Two kinds of Space, each with its own README under space/:

  static  (default) one HTML page that shows the recorded example answers (app/examples.json). It runs
          no model, costs nothing, and is free to host.
  docker  the Streamlit page itself, built from the Dockerfile. Hugging Face asks for a paid plan to
          host a Docker Space. With no Claude API key set on the Space it shows the recorded answers only.

Either way only the files that kind of Space needs are uploaded, as one commit, and the script waits
until the Space is running. It uses the Hugging Face login saved on this machine (`hf auth login`); no
token is read from or written to the repository. `--dry-run` lists the files without uploading anything.
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


def stage(dest: Path, sdk: str = "static", root: Path = ROOT) -> list[str]:
    """Copy what that kind of Space needs into dest and return the staged paths."""
    if sdk == "docker":
        for name in FILES:
            shutil.copy2(root / name, dest / name)
        for name in FOLDERS:
            shutil.copytree(root / name, dest / name, ignore=shutil.ignore_patterns(*SKIP))
    else:
        shutil.copy2(root / "space" / "static" / "index.html", dest / "index.html")
        shutil.copy2(root / "app" / "examples.json", dest / "examples.json")
        shutil.copy2(root / "LICENSE", dest / "LICENSE")
    shutil.copy2(root / "space" / sdk / "README.md", dest / "README.md")
    return sorted(path.relative_to(dest).as_posix() for path in dest.rglob("*") if path.is_file())


def revision(root: Path = ROOT) -> str:
    """The commit being deployed, marked when the working tree has changes that are not committed."""

    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout

    return git("rev-parse", "--short", "HEAD").strip() + ("+changes" if git("status", "--porcelain") else "")


def page_url(repo_id: str, sdk: str = "static") -> str:
    name = repo_id.lower().replace("/", "-").replace("_", "-").replace(".", "-")
    return f"https://{name}.static.hf.space" if sdk == "static" else f"https://{name}.hf.space"


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
    parser = argparse.ArgumentParser(description="Publish the demo as a Hugging Face Space.")
    parser.add_argument("repo_id", help="the Space, as owner/name")
    parser.add_argument("--sdk", choices=("static", "docker"), default="static", help="the kind of Space")
    parser.add_argument("--dry-run", action="store_true", help="list the files and upload nothing")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        files = stage(Path(tmp), args.sdk)
        print(f"{len(files)} files for the {args.sdk} Space:")
        for name in files:
            print(f"  {name}")
        if args.dry_run:
            return 0

        from huggingface_hub import HfApi  # imported here, so a dry run needs no extra package

        api = HfApi()
        api.create_repo(args.repo_id, repo_type="space", space_sdk=args.sdk, exist_ok=True)
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
    print(f"running at {page_url(args.repo_id, args.sdk)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
