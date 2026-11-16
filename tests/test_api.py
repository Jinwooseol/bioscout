"""The HTTP API, driven in-process: the real routes and the real agent over a scripted model and an
in-memory PubMed. Nothing listens on a port and nothing leaves the machine."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

pytest.importorskip("fastapi", reason="the api extra is not installed")

from fastapi.testclient import TestClient  # noqa: E402

from bioscout import __version__, api, demo  # noqa: E402
from bioscout.answer import Step, Usage  # noqa: E402
from bioscout.config import Settings  # noqa: E402
from bioscout.http import CacheMissError  # noqa: E402
from bioscout.quota import Quota  # noqa: E402
from tests.fakes import FakePubMed, reply  # noqa: E402
from tests.test_demo import QUOTE, TEXTS, canned_result, judged, scripted_llm  # noqa: E402

QUESTION = "Is X better than Y?"
ASK = {"question": QUESTION}
EXAMPLES = Path(__file__).parent.parent / "app" / "examples.json"
PAGE = "https://brakey-bioscout.static.hf.space"


async def agent(question: str, verify: bool = True, on_step: Any = None) -> demo.DemoResult:
    """The real agent, over a scripted model and an in-memory PubMed."""
    answer = reply("yes", ("X lowered mortality versus Y.", ["10000001"]))
    judgments = [judged("supported", ("10000001", "supported", QUOTE))]
    llm = scripted_llm(answer, judgments)
    return await demo.answer_question(llm, FakePubMed(texts=TEXTS), question, verify, on_step)


def build(
    folder: Path,
    answerer: demo.Answerer | None = agent,
    *,
    per_day: int = 5,
    total: int = 50,
    per_caller: int | None = None,
    recorded: dict[str, demo.DemoResult] | None = None,
    live: bool | None = None,
    **settings: Any,
):
    quota = Quota(folder / "quota.sqlite", per_day=per_day, total=total, per_caller=per_caller)
    return api.create_app(
        Settings(cache_dir=folder, **settings),
        answerer=answerer,
        quota=quota,
        recorded=recorded or {},
        live=live,
    )


def left_today(client: TestClient) -> int:
    return client.get("/v1/status").json()["quota"]["left_today"]


def events(text: str) -> list[tuple[str, dict[str, Any]]]:
    """The (event, data) pairs of a server-sent event stream; keep-alive comments are skipped."""
    seen = []
    for block in text.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines() if not line.startswith(":"))
        if fields:
            seen.append((fields["event"], json.loads(fields["data"])))
    return seen


def recorded_example() -> demo.DemoResult:
    example = canned_result()
    example.recorded = True
    return example


def test_a_question_gets_the_answer_with_verdicts_sources_and_what_it_cost(tmp_path: Path):
    with TestClient(build(tmp_path)) as client:
        response = client.post("/v1/answers", json={"question": f"  {QUESTION}\n"})
        assert response.status_code == 200
        body = response.json()
        assert body["question"] == QUESTION  # trimmed before the agent sees it
        assert body["decision"] == "yes" and not body["refused"]
        assert body["sentences"] == [
            {
                "text": "X lowered mortality versus Y.",
                "citations": ["10000001"],
                "verdict": "supported",
                "verdict_label": "Supported",
                "evidence": QUOTE,
            }
        ]
        assert body["sources"] == [
            {
                "pmid": "10000001",
                "url": "https://pubmed.ncbi.nlm.nih.gov/10000001/",
                "title": "Synthetic 10000001",
                "journal": None,
                "year": None,
                "sentences": [1],
            }
        ]
        assert [step["name"] for step in body["steps"]] == ["plan", "search", "screen", "answer", "verify"]
        assert body["steps"][0]["label"] == demo.STEP_LABELS["plan"]
        assert body["usd"] == pytest.approx(0.04) and body["seconds"] is not None  # four model calls
        assert body["verified"] and not body["recorded"] and not body["replayed"]
        assert body["searches"] and all(search["query"] for search in body["searches"])
        assert body["abstracts"] >= 1  # what the answer was written from
        assert left_today(client) == 4

        unchecked = client.post("/v1/answers", json={"question": QUESTION, "verify": False}).json()
        assert not unchecked["verified"]
        assert [s["verdict"] for s in unchecked["sentences"]] == [None]
        assert unchecked["sentences"][0]["verdict_label"] == "Not checked"
        assert [step["name"] for step in unchecked["steps"]] == ["plan", "search", "screen", "answer"]


def test_the_stream_reports_each_step_as_it_finishes_and_ends_with_the_same_answer(tmp_path: Path):
    with TestClient(build(tmp_path)) as client:
        response = client.post("/v1/answers/stream", json=ASK)
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        seen = events(response.text)
        assert [name for name, _ in seen] == ["step"] * 5 + ["answer"]
        assert [data["name"] for _, data in seen[:-1]] == ["plan", "search", "screen", "answer", "verify"]
        assert seen[0][1]["label"] == demo.STEP_LABELS["plan"] and "seconds" in seen[0][1]

        streamed, plain = seen[-1][1], client.post("/v1/answers", json=ASK).json()
        assert set(streamed) == set(plain)
        for part in ("question", "decision", "sentences", "sources", "dropped", "verified", "recorded"):
            assert streamed[part] == plain[part]
        assert left_today(client) == 3  # either way of asking counts one question


def test_an_example_question_returns_its_recorded_answer_and_uses_no_allowance(tmp_path: Path):
    calls: list[bool] = []

    async def counting(question: str, verify: bool = True, on_step: Any = None) -> demo.DemoResult:
        calls.append(verify)
        return await agent(question, verify, on_step)

    with TestClient(build(tmp_path, counting, recorded={QUESTION: recorded_example()})) as client:
        body = client.post("/v1/answers", json=ASK).json()
        assert body["recorded"] and body["replayed"] and body["verified"]
        assert body["usd"] == 0.1 and body["seconds"] is None  # its cost when recorded; no real timing
        assert body["dropped"][0]["explanation"] == "Still not backed after one rewrite."
        assert body["sources"][1]["title"] is None  # a cited paper the agent never fetched

        seen = events(client.post("/v1/answers/stream", json=ASK).text)
        assert [name for name, _ in seen] == ["answer"] and seen[0][1] == body
        assert client.get("/v1/examples").json() == [body]
        assert not calls and left_today(client) == 5

        # A recorded answer went through the citation check; asked for without it, the agent runs.
        unchecked = client.post("/v1/answers", json={"question": QUESTION, "verify": False}).json()
        assert not unchecked["recorded"] and calls == [False] and left_today(client) == 4


def test_the_allowance_is_one_for_every_caller_and_a_used_up_one_answers_429(tmp_path: Path):
    with TestClient(build(tmp_path, per_day=1)) as client:
        assert client.post("/v1/answers", json=ASK).status_code == 200
        refused = client.post("/v1/answers", json=ASK)
        assert refused.status_code == 429
        detail = refused.json()["detail"]
        assert detail["code"] == "quota_exceeded" and "Today's allowance" in detail["message"]
        assert 0 < int(refused.headers["retry-after"]) <= 86400  # until the next UTC midnight
        # The stream refuses the same way: an ordinary error response, before any event.
        stream = client.post("/v1/answers/stream", json=ASK)
        assert stream.status_code == 429 and stream.headers["content-type"] == "application/json"

    with TestClient(build(tmp_path, per_day=1)) as restarted:  # the count is in the file
        assert restarted.post("/v1/answers", json=ASK).status_code == 429
        assert left_today(restarted) == 0

    with TestClient(build(tmp_path, per_day=5, total=1)) as client:  # the total: waiting does not help
        refused = client.post("/v1/answers", json=ASK)
        assert refused.status_code == 429 and "retry-after" not in refused.headers
        assert "for this deployment is used up" in refused.json()["detail"]["message"]


def test_one_caller_may_ask_only_so_many_of_the_days_questions(tmp_path: Path):
    app = build(tmp_path, per_day=10, per_caller=2, api_client_header="Fly-Client-IP")

    def ask(address: str | None):
        headers = {"Fly-Client-IP": address} if address else {}
        return client.post("/v1/answers", json=ASK, headers=headers)

    with TestClient(app) as client:
        assert client.get("/v1/status").json()["limits"]["questions_per_caller"] == 2
        assert [ask("203.0.113.9").status_code for _ in range(3)] == [200, 200, 429]
        refused = ask("203.0.113.9")
        detail = refused.json()["detail"]
        assert detail["code"] == "quota_exceeded" and "one visitor may ask in a day" in detail["message"]
        assert 0 < int(refused.headers["retry-after"]) <= 86400
        assert left_today(client) == 8  # a refused question is not counted

        assert ask("203.0.113.10").status_code == 200  # another caller still has a share
        # One IPv6 subscriber holds a whole /64, so its addresses are one caller.
        assert [ask(f"2001:db8:1:2::{n}").status_code for n in (1, 2, 3)] == [200, 200, 429]
        assert ask("2001:db8:1:3::1").status_code == 200
        # With no such header on the request, the caller is where the connection comes from.
        assert [ask(None).status_code for _ in range(3)] == [200, 200, 429]
        assert left_today(client) == 2

    # Without the setting a header a caller sends is not believed: all of these are one caller.
    with TestClient(build(tmp_path / "plain", per_day=10, per_caller=1)) as client:
        assert [ask(address).status_code for address in ("203.0.113.9", "203.0.113.10")] == [200, 429]


def test_without_a_key_only_the_recorded_answers_are_served(tmp_path: Path):
    app = build(tmp_path, answerer=None, recorded={QUESTION: recorded_example()}, live=False)
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok", "version": __version__}
        status = client.get("/v1/status").json()
        assert not status["live"] and status["examples"] == 1
        assert status["limits"] == {
            "question_chars": 400,
            "questions_per_day": 5,
            "questions_total": 50,
            "questions_per_caller": 5,
            "concurrent": 2,
            "timeout_s": 300.0,
        }
        assert client.post("/v1/answers", json=ASK).json()["recorded"]

        refused = client.post("/v1/answers", json={"question": "Something else?"})
        assert refused.status_code == 503 and refused.json()["detail"]["code"] == "live_off"
        assert client.post("/v1/answers/stream", json={"question": "Something else?"}).status_code == 503
        assert left_today(client) == 5


async def test_only_so_many_answers_run_at_once_and_a_busy_refusal_uses_no_allowance(tmp_path: Path):
    started, release = asyncio.Event(), asyncio.Event()

    async def slow(question: str, verify: bool = True, on_step: Any = None) -> demo.DemoResult:
        started.set()
        await release.wait()
        return canned_result()

    app = build(tmp_path, slow, api_max_concurrent=1)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        first = asyncio.create_task(client.post("/v1/answers", json=ASK))
        await started.wait()
        busy = await client.post("/v1/answers", json=ASK)
        assert busy.status_code == 503 and busy.json()["detail"]["code"] == "busy"
        assert busy.headers["retry-after"] == "30"

        release.set()
        assert (await first).status_code == 200
        assert (await client.post("/v1/answers", json=ASK)).status_code == 200  # the slot is free again
        status = (await client.get("/v1/status")).json()
        assert status["quota"]["left_today"] == 3  # two answered; the busy one was not counted


def test_a_failed_answer_names_the_kind_of_error_and_never_quotes_it(tmp_path: Path, caplog):
    request_text = "https://eutils.example/esearch.fcgi?term=x&api_key=SECRETKEY&email=someone@example.com"

    async def broken(question: str, verify: bool = True, on_step: Any = None) -> demo.DemoResult:
        raise RuntimeError(request_text)

    with caplog.at_level(logging.INFO, logger="bioscout.api"), TestClient(build(tmp_path, broken)) as client:
        failed = client.post("/v1/answers", json=ASK)
        assert failed.status_code == 502
        assert failed.json()["detail"] == {
            "code": "answer_failed",
            "message": "The answer could not be produced (RuntimeError).",
        }
        streamed = client.post("/v1/answers/stream", json=ASK)
        assert events(streamed.text) == [("error", failed.json()["detail"])]
        assert left_today(client) == 3  # a question that failed stays counted

    assert "RuntimeError" in caplog.text and "broken" in caplog.text  # the kind, and where it happened
    for secret in ("SECRETKEY", "someone@example.com"):
        assert secret not in caplog.text and secret not in failed.text and secret not in streamed.text


def test_a_replay_only_server_says_what_is_not_recorded_and_a_slow_answer_is_stopped(tmp_path: Path):
    async def not_recorded(question: str, verify: bool = True, on_step: Any = None) -> demo.DemoResult:
        raise CacheMissError("No recorded response for https://eutils.example/esearch.fcgi {'term': 'x'}")

    with TestClient(build(tmp_path, not_recorded)) as client:
        missed = client.post("/v1/answers", json=ASK)
        assert missed.status_code == 404 and missed.json()["detail"]["code"] == "not_recorded"

    async def endless(question: str, verify: bool = True, on_step: Any = None) -> demo.DemoResult:
        await asyncio.sleep(60)
        raise AssertionError("the time limit should have stopped this")

    with TestClient(build(tmp_path, endless, api_timeout_s=0.05)) as client:
        late = client.post("/v1/answers", json=ASK)
        assert late.status_code == 504 and late.json()["detail"]["code"] == "timeout"


async def test_closing_the_stream_stops_the_agent(tmp_path: Path):
    stopped = asyncio.Event()

    async def slow(question: str, verify: bool = True, on_step: Any = None) -> demo.DemoResult:
        on_step(Step(name="plan", usage=Usage(latency_ms=1200)))
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            stopped.set()
            raise
        raise AssertionError("the caller left; the agent should have been stopped")

    request = SimpleNamespace(app=build(tmp_path, slow))
    stream = api.answer_stream(request, api.Ticket(api.Ask(question=QUESTION)))
    first = await anext(stream)
    assert first.event == "step" and (first.data.name, first.data.seconds) == ("plan", 1.2)
    await stream.aclose()  # what the server does when the caller disconnects
    await asyncio.wait_for(stopped.wait(), timeout=2)


def test_a_question_must_be_text_of_a_sensible_length(tmp_path: Path):
    calls: list[str] = []

    async def counting(question: str, verify: bool = True, on_step: Any = None) -> demo.DemoResult:
        calls.append(question)
        return canned_result()

    bad = [{"question": ""}, {"question": "   "}, {"question": "x" * 401}, {}, {"question": 5}]
    with TestClient(build(tmp_path, counting)) as client:
        for body in bad:
            assert client.post("/v1/answers", json=body).status_code == 422
            assert client.post("/v1/answers/stream", json=body).status_code == 422
        assert not calls and left_today(client) == 5  # refused before anything is counted or run
        assert client.post("/v1/answers", json={"question": "x" * 400}).status_code == 200


def test_a_body_larger_than_any_question_is_refused_before_it_is_read(tmp_path: Path):
    calls: list[str] = []

    async def counting(question: str, verify: bool = True, on_step: Any = None) -> demo.DemoResult:
        calls.append(question)
        return canned_result()

    def unsized():  # sent in chunks, with no Content-Length
        yield json.dumps(ASK).encode()

    json_body = {"Content-Type": "application/json"}
    with TestClient(build(tmp_path, counting, api_cors_origins=(PAGE,))) as client:
        large = json.dumps({"question": "x" * api.MAX_BODY_BYTES}).encode()
        for path in ("/v1/answers", "/v1/answers/stream"):
            refused = client.post(path, content=large, headers={**json_body, "Origin": PAGE})
            assert refused.status_code == 413 and refused.json()["detail"]["code"] == "too_large"
            assert refused.headers["access-control-allow-origin"] == PAGE  # a page can read why
            chunked = client.post(path, content=unsized(), headers=json_body)
            assert chunked.status_code == 411 and chunked.json()["detail"]["code"] == "length_required"
        assert not calls and left_today(client) == 5
        assert client.post("/v1/answers", json=ASK).status_code == 200  # an ordinary question passes


def test_the_address_of_the_server_itself_leads_to_the_docs(tmp_path: Path):
    with TestClient(build(tmp_path)) as client:
        home = client.get("/", follow_redirects=False)
        assert home.status_code == 307 and home.headers["location"] == "/docs"
        assert client.get("/docs").status_code == 200


def test_the_schema_lists_the_routes_and_how_a_question_is_refused(tmp_path: Path):
    schema = build(tmp_path).openapi()
    assert set(schema["paths"]) == {
        "/health", "/v1/status", "/v1/examples", "/v1/answers", "/v1/answers/stream",
    }  # fmt: skip
    answers = schema["paths"]["/v1/answers"]["post"]["responses"]
    assert {"200", "404", "422", "429", "502", "503", "504"} <= set(answers)
    stream = schema["paths"]["/v1/answers/stream"]["post"]["responses"]
    assert "text/event-stream" in stream["200"]["content"] and {"429", "503"} <= set(stream)
    assert schema["components"]["schemas"]["Ask"]["properties"]["question"]["maxLength"] == 400


def test_a_page_on_another_origin_may_call_only_when_it_is_listed(tmp_path: Path, monkeypatch):
    with TestClient(build(tmp_path, api_cors_origins=(PAGE,))) as client:
        assert client.get("/health", headers={"Origin": PAGE}).headers["access-control-allow-origin"] == PAGE
        other = client.get("/health", headers={"Origin": "https://elsewhere.example"})
        assert "access-control-allow-origin" not in other.headers
        asking = {"Origin": PAGE, "Access-Control-Request-Method": "POST"}
        preflight = client.options("/v1/answers", headers=asking)
        assert preflight.status_code == 200 and "POST" in preflight.headers["access-control-allow-methods"]

    with TestClient(build(tmp_path)) as client:  # the default: no other origin
        assert "access-control-allow-origin" not in client.get("/health", headers={"Origin": PAGE}).headers

    monkeypatch.setenv("BIOSCOUT_API_CORS_ORIGINS", f"{PAGE}/, https://b.example")
    assert Settings.from_env(dotenv=None).api_cors_origins == (PAGE, "https://b.example")


def test_the_bundled_examples_are_served_without_a_key_and_hold_no_abstract_passage(tmp_path: Path):
    settings = Settings(cache_dir=tmp_path, examples_path=EXAMPLES)
    app = api.create_app(settings, quota=Quota(tmp_path / "quota.sqlite", 5, 50), live=False)
    with TestClient(app) as client:
        served = client.get("/v1/examples").json()
        assert [example["question"] for example in served] == demo.EXAMPLE_QUESTIONS
        assert all(example["recorded"] and example["sentences"] for example in served)
        assert all(s["evidence"] is None for example in served for s in example["sentences"])
        again = client.post("/v1/answers", json={"question": demo.EXAMPLE_QUESTIONS[0]}).json()
        assert again == served[0]


def test_the_server_opens_the_agents_clients_once_and_closes_them_when_it_stops(tmp_path: Path, monkeypatch):
    seen: list[Any] = []

    async def stub(llm: Any, pubmed: Any, question: str, verify: bool, on_step: Any) -> demo.DemoResult:
        seen.append((llm, pubmed))
        return canned_result()

    monkeypatch.setattr(demo, "answer_question", stub)
    settings = Settings(cache_dir=tmp_path)  # no tracing keys: nothing is wrapped
    app = api.create_app(settings, quota=Quota(tmp_path / "quota.sqlite", 5, 50), recorded={}, live=True)
    assert app.state.service.answerer is None  # nothing is opened until the server starts
    with TestClient(app) as client:
        assert client.get("/v1/status").json()["live"]
        for _ in range(2):
            assert client.post("/v1/answers", json=ASK).status_code == 200
    # Both questions went through the same clients, so they share one PubMed rate limit.
    assert len(seen) == 2 and seen[0][0] is seen[1][0] and seen[0][1] is seen[1][1]
    assert app.state.service.answerer is None
