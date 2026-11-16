"""What goes to the Hugging Face Space: only what that kind of Space needs, and nothing secret or local."""

from __future__ import annotations

import importlib.util
import json
import re
import tomllib
from pathlib import Path

import pytest

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
    assert files == ["LICENSE", "README.md", "config.json", "examples.json", "index.html"]
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


def test_the_page_draws_a_live_answer_from_what_the_api_returns():
    pytest.importorskip("fastapi", reason="the api extra is not installed")
    from bioscout import api

    page = (ROOT / "space" / "static" / "index.html").read_text(encoding="utf-8")
    assert 'getJson("config.json"' in page
    assert "/v1/status" in page and "/v1/answers/stream" in page
    # Every part of an answer the page reads is a part the API sends, and a recorded answer is put
    # in that same shape before it is drawn.
    read = set(re.findall(r"\bview\.(\w+)", page))
    assert {"sentences", "sources", "steps", "abstracts", "usd", "replayed"} <= read
    assert read <= set(api.AnswerOut.model_fields)
    assert set(re.findall(r"\bsource\.(\w+)", page)) <= set(api.SourceOut.model_fields)
    assert set(re.findall(r"\bsentence\.(\w+)", page)) <= set(api.SentenceOut.model_fields)
    assert set(re.findall(r"\bstatus\.(\w+)", page)) <= set(api.StatusOut.model_fields)
    assert set(re.findall(r"\bstatus\.limits\.(\w+)", page)) <= set(api.LimitsOut.model_fields)
    assert set(re.findall(r"\bstatus\.quota\.(\w+)", page)) <= set(api.QuotaOut.model_fields)


def test_the_page_asks_only_a_plain_https_address_and_the_api_lets_only_that_page_in(tmp_path: Path):
    deploy = deploy_script()
    config = tmp_path / "config.json"

    def address(value: object) -> str:
        config.write_text(json.dumps({"api": value}), encoding="utf-8")
        return deploy.api_address(config)

    assert address("") == "" and address(None) == ""  # no API: the recorded answers only
    assert address("https://bioscout-api.fly.dev/") == "https://bioscout-api.fly.dev"
    bad = (
        "http://bioscout-api.fly.dev",  # a question would travel in the clear
        "https://bioscout-api.fly.dev/v1",
        "https://bioscout-api.fly.dev/?key=1",
        "https://someone:secret@bioscout-api.fly.dev",
        "bioscout-api.fly.dev",
    )
    for value in bad:
        with pytest.raises(ValueError):
            address(value)

    # The committed file is publishable as it is, and a bad one stops the upload before it starts.
    deploy.api_address(ROOT / "space" / "static" / "config.json")
    broken = tmp_path / "repo"
    (broken / "space" / "static").mkdir(parents=True)
    (broken / "space" / "static" / "config.json").write_text('{"api": "http://x.example"}', encoding="utf-8")
    with pytest.raises(ValueError):
        deploy.stage(tmp_path / "out", root=broken)

    # The one origin the hosted API accepts browser calls from is this page's own address.
    fly = tomllib.loads((ROOT / "fly.toml").read_text(encoding="utf-8"))
    allowed = fly["env"]["BIOSCOUT_API_CORS_ORIGINS"]
    assert allowed.endswith(".static.hf.space") and "," not in allowed
    assert allowed == deploy.page_url(allowed.removeprefix("https://").removesuffix(".static.hf.space"))


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
