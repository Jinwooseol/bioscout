"""What goes to the Hugging Face Space: only what the image needs, and nothing secret or local."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent


def deploy_script():
    spec = importlib.util.spec_from_file_location("deploy_space", ROOT / "scripts" / "deploy_space.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_space_gets_what_the_dockerfile_copies_and_nothing_else(tmp_path: Path):
    deploy = deploy_script()
    files = deploy.stage(tmp_path)

    assert {"Dockerfile", "uv.lock", "app/streamlit_app.py", "app/examples.json"} <= set(files)
    assert "src/bioscout/demo.py" in files
    assert {name.split("/")[0] for name in files} == {*deploy.FILES, *deploy.FOLDERS, "README.md"}
    assert not [name for name in files if ".env" in name or name.endswith((".sqlite", ".pyc"))]

    # Every path the Dockerfile copies is in the upload, so the Space can build the image.
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    copied = re.findall(r"^COPY --chown=user:user (.+) \S+$", dockerfile, flags=re.M)
    for name in " ".join(copied).split():
        assert (tmp_path / name).exists(), name


def test_the_space_readme_declares_a_docker_space_on_the_port_the_image_serves(tmp_path: Path):
    deploy_script().stage(tmp_path)
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    front_matter = readme.split("---")[1]
    assert "sdk: docker" in front_matter and "app_port: 7860" in front_matter
    assert "EXPOSE 7860" in (ROOT / "Dockerfile").read_text(encoding="utf-8")
    summary = re.search(r"^short_description: (.+)$", front_matter, flags=re.M).group(1)
    assert len(summary) <= 60  # the limit Hugging Face puts on a Space's short description


def test_the_page_address_follows_the_space_name():
    assert deploy_script().page_url("Some_Owner/My.Space") == "https://some-owner-my-space.hf.space"
