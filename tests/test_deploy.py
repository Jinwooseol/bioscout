"""What goes to the Hugging Face Space: only what that kind of Space needs, and nothing secret or local."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

from bioscout import demo

ROOT = Path(__file__).parent.parent


def deploy_script():
    spec = importlib.util.spec_from_file_location("deploy_space", ROOT / "scripts" / "deploy_space.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def front_matter(readme: Path) -> str:
    return readme.read_text(encoding="utf-8").split("---")[1]


def test_the_static_space_is_one_page_and_the_recorded_answers(tmp_path: Path):
    files = deploy_script().stage(tmp_path)
    assert files == ["LICENSE", "README.md", "examples.json", "index.html"]
    assert "sdk: static" in front_matter(tmp_path / "README.md")
    # The page reads the same recorded answers as the Streamlit page, which hold no article text.
    assert (tmp_path / "examples.json").read_bytes() == (ROOT / "app" / "examples.json").read_bytes()
    recorded = json.loads((tmp_path / "examples.json").read_text(encoding="utf-8"))
    assert [record["answer"]["question"] for record in recorded] == list(demo.EXAMPLE_QUESTIONS)


def test_the_static_page_uses_the_same_labels_as_the_streamlit_page():
    page = (ROOT / "space" / "static" / "index.html").read_text(encoding="utf-8")
    labels = [*demo.VERDICT_LABELS.values(), *demo.DROP_REASONS.values(), *demo.STEP_LABELS.values()]
    assert not [label for label in labels if f'"{label}"' not in page]
    assert 'fetch("examples.json")' in page and demo.REPO_URL in page
    assert "innerHTML" not in page  # recorded text is only ever set as text


def test_the_docker_space_gets_what_the_dockerfile_copies_and_nothing_else(tmp_path: Path):
    deploy = deploy_script()
    files = deploy.stage(tmp_path, "docker")

    assert {"Dockerfile", "uv.lock", "app/streamlit_app.py", "app/examples.json"} <= set(files)
    assert "src/bioscout/demo.py" in files
    assert {name.split("/")[0] for name in files} == {*deploy.FILES, *deploy.FOLDERS, "README.md"}
    assert not [name for name in files if ".env" in name or name.endswith((".sqlite", ".pyc"))]

    # Every path the Dockerfile copies is in the upload, so the Space can build the image.
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    copied = re.findall(r"^COPY --chown=user:user (.+) \S+$", dockerfile, flags=re.M)
    for name in " ".join(copied).split():
        assert (tmp_path / name).exists(), name

    matter = front_matter(tmp_path / "README.md")
    assert "sdk: docker" in matter and "app_port: 7860" in matter and "EXPOSE 7860" in dockerfile


def test_both_space_descriptions_fit_the_limit():
    for sdk in ("static", "docker"):
        matter = front_matter(ROOT / "space" / sdk / "README.md")
        summary = re.search(r"^short_description: (.+)$", matter, flags=re.M).group(1)
        assert len(summary) <= 60  # the limit Hugging Face puts on a Space's short description


def test_the_page_address_follows_the_space_name_and_kind():
    deploy = deploy_script()
    assert deploy.page_url("Some_Owner/My.Space") == "https://some-owner-my-space.static.hf.space"
    assert deploy.page_url("Some_Owner/My.Space", "docker") == "https://some-owner-my-space.hf.space"
