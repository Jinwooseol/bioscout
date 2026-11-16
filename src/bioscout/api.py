"""BioScout over HTTP: ask a question and get the cited answer as JSON, or follow the agent step by
step as server-sent events.

    uv run --extra api bioscout-api        # http://127.0.0.1:8000, interactive docs at /docs

    GET  /health               the process is up
    GET  /v1/status            whether new questions can be answered, the limits, the allowance left
    GET  /v1/examples          the recorded answers to the example questions: free, no model call
    POST /v1/answers           {"question": "...", "verify": true} -> the answer
    POST /v1/answers/stream    the same as events: step, step, ..., then answer (or error)

A question that runs the agent is paid for with the operator's Claude API key. Before it starts it
passes four checks, in this order, and only the last one counts it:

  1. a recorded answer is returned as it is (the example questions cost nothing);
  2. with no key to run with, the request is refused (503 live_off);
  3. a free slot: BIOSCOUT_API_MAX_CONCURRENT answers run at a time (503 busy);
  4. the question allowance of quota.py: per day and in total for everyone, and per day for the
     caller (429 quota_exceeded).

The layout of an answer comes from demo.py, which the demo page shares, so both show the same thing.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import traceback
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel, Field, StringConstraints
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Receive, Scope, Send

from bioscout import __version__, demo, tracing
from bioscout.answer import Decision, Step, Verdict
from bioscout.config import Settings
from bioscout.http import CacheMissError
from bioscout.quota import Quota, QuotaExceeded

log = logging.getLogger("bioscout.api")

BUSY_RETRY_S = 30
MAX_BODY_BYTES = 16 * 1024  # a question is at most 400 characters; no request needs a larger body
LIVE_OFF = (
    "Live questions are switched off on this server: no Claude API key is configured. "
    "GET /v1/examples returns the recorded answers."
)
BUSY = "The server is answering as many questions as it takes at a time. Try again shortly."
NOT_RECORDED = "This server replays recorded responses only, and this question was not recorded."
DESCRIPTION = (
    "Ask a biomedical research question. BioScout searches PubMed, answers with a citation on every "
    "sentence, and checks each citation against the paper's abstract.\n\n"
    "BioScout helps you find and read biomedical literature. It is not a medical device and must not "
    "be used to make diagnostic or treatment decisions about individual patients."
)

# ---------------------------------------------------------------------------- what goes in and out


class Ask(BaseModel):
    question: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=demo.MAX_QUESTION_CHARS)
    ] = Field(description="A biomedical research question, in English.")
    verify: bool = Field(
        default=True,
        description="Judge each sentence against the abstracts it cites, rewrite once what is not fully "
        "backed, and remove what nothing backs. Off: the answer as first written.",
    )


class SentenceOut(BaseModel):
    text: str
    citations: list[str] = Field(description="PMIDs, each among the abstracts the answer was written from")
    verdict: Verdict | None = Field(description="The citation check's verdict; null when it did not run")
    verdict_label: str
    evidence: str | None = Field(description="The passage of the abstract the verdict rests on")


class DroppedOut(BaseModel):
    text: str
    reason: str
    explanation: str


class SourceOut(BaseModel):
    pmid: str
    url: str
    title: str | None = Field(description="null: the agent never fetched a paper with this PMID")
    journal: str | None
    year: int | None
    sentences: list[int] = Field(description="The sentences, numbered from 1, that cite this paper")


class SearchOut(BaseModel):
    query: str
    total_found: int


class StepOut(BaseModel):
    name: str
    label: str
    runs: int
    llm_calls: int
    usd: float
    seconds: float


class AnswerOut(BaseModel):
    question: str
    decision: Decision | None
    refused: bool = Field(description="The model declined to answer")
    sentences: list[SentenceOut]
    dropped: list[DroppedOut] = Field(description="Sentences the citation check removed, and why")
    sources: list[SourceOut] = Field(description="The cited papers, in the order they are first cited")
    searches: list[SearchOut]
    abstracts: int = Field(description="How many abstracts the answer was written from")
    steps: list[StepOut]
    usd: float = Field(description="What the answer cost when it was produced")
    seconds: float | None = Field(description="null for an answer replayed from recorded responses")
    verified: bool = Field(description="The citation check ran")
    recorded: bool = Field(description="A recorded answer to an example question, not produced just now")
    replayed: bool = Field(description="Every model call came from recorded responses")


class StepEvent(BaseModel):
    name: str
    label: str
    seconds: float


class LimitsOut(BaseModel):
    question_chars: int
    questions_per_day: int
    questions_total: int
    questions_per_caller: int = Field(description="Of the day's questions, how many one caller may ask")
    concurrent: int
    timeout_s: float


class QuotaOut(BaseModel):
    left_today: int
    left_total: int
    resets_at: datetime


class StatusOut(BaseModel):
    live: bool = Field(description="New questions can be answered; otherwise only the recorded examples")
    model: str
    examples: int
    limits: LimitsOut
    quota: QuotaOut


class HealthOut(BaseModel):
    status: Literal["ok"] = "ok"
    version: str


class ProblemOut(BaseModel):
    code: str
    message: str


class ProblemBody(BaseModel):
    detail: ProblemOut


class Problem(Exception):
    """A request that cannot be served, as the status and body the caller gets."""

    def __init__(self, status: int, code: str, message: str, retry_after_s: int | None = None) -> None:
        super().__init__(code)
        self.status = status
        self.code = code
        self.message = message
        self.retry_after_s = retry_after_s

    def body(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message}


def answer_out(result: demo.DemoResult, verified: bool) -> AnswerOut:
    answer = result.answer
    usd, seconds = demo.totals(answer)
    replayed = result.recorded or demo.was_replayed(answer)
    sources = [
        SourceOut(
            pmid=source.pmid,
            url=source.url,
            title=source.article.title if source.article else None,
            journal=(source.article.journal or None) if source.article else None,
            year=source.article.year if source.article else None,
            sentences=source.sentences,
        )
        for source in demo.sources_of(result)
    ]
    return AnswerOut(
        question=answer.question,
        decision=answer.decision,
        refused=answer.refused,
        sentences=[
            SentenceOut(
                text=s.text,
                citations=s.citations,
                verdict=s.verdict,
                verdict_label=demo.VERDICT_LABELS[s.verdict],
                evidence=s.evidence,
            )
            for s in answer.sentences
        ],
        dropped=[
            DroppedOut(text=d.text, reason=d.reason, explanation=demo.DROP_REASONS.get(d.reason, d.reason))
            for d in answer.dropped
        ],
        sources=sources,
        searches=[SearchOut(query=s.query, total_found=s.total_found) for s in answer.searches],
        abstracts=len(answer.context_pmids),
        steps=[StepOut(**vars(total)) for total in demo.step_totals(answer)],
        usd=round(usd, 4),
        seconds=None if replayed else round(seconds, 1),
        verified=verified,
        recorded=result.recorded,
        replayed=replayed,
    )


def step_event(step: Step) -> ServerSentEvent:
    label = demo.STEP_LABELS.get(step.name, step.name)
    event = StepEvent(name=step.name, label=label, seconds=round(step.usage.latency_ms / 1000, 1))
    return ServerSentEvent(data=event, event="step")


# ------------------------------------------------------------------------------- letting a question in


@dataclass
class Service:
    """What the routes work with. Tests build it with a scripted answerer and a quota in a temp folder."""

    settings: Settings
    quota: Quota
    recorded: Mapping[str, demo.DemoResult]
    live: bool  # there is a key (or a cache) to answer new questions with
    answerer: demo.Answerer | None = None  # None until the app starts, and for good when live is off
    running: int = 0  # questions being answered now


@dataclass
class Ticket:
    """A question that was let in: either its recorded answer, or the right to run the agent once."""

    ask: Ask
    recorded: demo.DemoResult | None = None


def service_of(request: Request) -> Service:
    return request.app.state.service


def caller_of(request: Request, header: str | None = None) -> str | None:
    """What tells one caller from another, for the limit per caller: the network address.

    Behind a proxy the connection comes from the proxy, so `header` names the request header in which
    that proxy gives the caller's address. It must be one the proxy sets itself (Fly-Client-IP on
    Fly.io), not one a caller can fill in, like X-Forwarded-For. An IPv6 caller is counted by /64: one
    subscriber holds that whole range."""
    given = request.headers.get(header) if header else None
    raw = (given or (request.client.host if request.client else "")).strip()
    if not raw:
        return None
    try:
        address = ipaddress.ip_address(raw)
    except ValueError:
        return raw[:100]  # not an address (a test client's name): still tells callers apart
    if isinstance(address, ipaddress.IPv6Address):
        if address.ipv4_mapped is not None:
            return str(address.ipv4_mapped)
        return str(ipaddress.ip_network(f"{address}/64", strict=False))
    return str(address)


async def admit(ask: Ask, request: Request) -> AsyncIterator[Ticket]:
    """The four checks of the module docstring. The slot is held until the response has been sent,
    which for a stream is after its last event."""
    service = service_of(request)
    # A recorded answer was checked, so it stands in only for a request that asks for the check.
    recorded = service.recorded.get(ask.question) if ask.verify else None
    if recorded is not None:
        yield Ticket(ask, recorded)
        return
    if not service.live or service.answerer is None:
        raise Problem(503, "live_off", LIVE_OFF)
    if service.running >= service.settings.api_max_concurrent:
        raise Problem(503, "busy", BUSY, retry_after_s=BUSY_RETRY_S)
    try:
        service.quota.take(caller_of(request, service.settings.api_client_header))
    except QuotaExceeded as exc:
        raise Problem(429, "quota_exceeded", demo.quota_text(exc), exc.retry_after_s) from None
    service.running += 1
    try:
        yield Ticket(ask)
    finally:
        service.running -= 1


async def answer_for(
    service: Service, ticket: Ticket, on_step: Callable[[Step], None] | None = None
) -> AnswerOut:
    if ticket.recorded is not None:
        return answer_out(ticket.recorded, verified=True)
    ask, limit = ticket.ask, service.settings.api_timeout_s
    assert service.answerer is not None  # admit() checked
    try:
        async with asyncio.timeout(limit):
            result = await service.answerer(ask.question, ask.verify, on_step)
    except TimeoutError:
        log.warning("answer stopped after %.0f s", limit)
        raise Problem(504, "timeout", f"The answer took longer than {limit:.0f} seconds.") from None
    except CacheMissError:
        raise Problem(404, "not_recorded", NOT_RECORDED) from None
    except Exception as exc:
        # The kind of error and where it happened, never its text: an exception can quote a request,
        # and a request to PubMed carries the API key and the contact address.
        frames = "".join(traceback.format_tb(exc.__traceback__))
        log.error("answer failed: %s\n%s", type(exc).__name__, frames)
        raise Problem(502, "answer_failed", demo.failure_text(exc)) from None
    out = answer_out(result, verified=ask.verify)
    how = "replayed, first cost" if out.replayed else "cost"
    log.info("answered: %d characters, verify=%s, %s $%.3f", len(ask.question), ask.verify, how, out.usd)
    return out


# ------------------------------------------------------------------------------------------ the routes

router = APIRouter()
Admitted = Annotated[Ticket, Depends(admit)]
REFUSALS: dict[int | str, dict[str, object]] = {
    429: {"model": ProblemBody, "description": "An allowance of questions is used up (`quota_exceeded`)"},
    503: {"model": ProblemBody, "description": "No key to answer with (`live_off`), or `busy`"},
}
FAILURES: dict[int | str, dict[str, object]] = {
    404: {"model": ProblemBody, "description": "Replay-only server, question not recorded"},
    502: {"model": ProblemBody, "description": "The agent failed (`answer_failed`)"},
    504: {"model": ProblemBody, "description": "The answer took too long (`timeout`)"},
}


@router.get("/health", tags=["service"], summary="The process is up")
async def health() -> HealthOut:
    return HealthOut(version=__version__)


@router.get("/v1/status", tags=["service"], summary="Can it answer, the limits, the allowance left")
async def status(request: Request) -> StatusOut:
    service = service_of(request)
    settings, quota = service.settings, service.quota.status()
    return StatusOut(
        live=service.live and service.answerer is not None,
        model=settings.llm_model,
        examples=len(service.recorded),
        limits=LimitsOut(
            question_chars=demo.MAX_QUESTION_CHARS,
            questions_per_day=quota.per_day,
            questions_total=quota.total,
            questions_per_caller=quota.per_caller,
            concurrent=settings.api_max_concurrent,
            timeout_s=settings.api_timeout_s,
        ),
        quota=QuotaOut(left_today=quota.left_today, left_total=quota.left_total, resets_at=quota.resets_at),
    )


@router.get("/v1/examples", tags=["answers"], summary="The recorded answers to the example questions")
async def examples(request: Request) -> list[AnswerOut]:
    return [answer_out(result, verified=True) for result in service_of(request).recorded.values()]


@router.post(
    "/v1/answers",
    tags=["answers"],
    summary="Answer a question",
    responses={**REFUSALS, **FAILURES},
)
async def answer(request: Request, ticket: Admitted) -> AnswerOut:
    """Runs the agent and returns when the answer is complete, which takes from twenty seconds to a
    few minutes. An example question returns its recorded answer at once."""
    return await answer_for(service_of(request), ticket)


@router.post(
    "/v1/answers/stream",
    tags=["answers"],
    summary="Answer a question, reporting each step as it finishes",
    response_class=EventSourceResponse,
    responses=REFUSALS,
)
async def answer_stream(request: Request, ticket: Admitted) -> AsyncIterator[ServerSentEvent]:
    """Server-sent events: a `step` event as each step of the agent finishes, then one `answer` event
    carrying the same body as `POST /v1/answers`, or one `error` event (`code`, `message`). A question
    that is not let in gets an ordinary error response instead of a stream. Closing the connection
    stops the agent."""
    service = service_of(request)
    events: asyncio.Queue[ServerSentEvent | None] = asyncio.Queue()

    async def work() -> None:
        try:
            out = await answer_for(service, ticket, lambda step: events.put_nowait(step_event(step)))
            events.put_nowait(ServerSentEvent(data=out, event="answer"))
        except Problem as problem:
            events.put_nowait(ServerSentEvent(data=problem.body(), event="error"))
        except Exception as exc:  # a fault of ours after the agent finished: still end with an event
            log.error("stream failed: %s", type(exc).__name__)
            body = Problem(500, "internal", demo.failure_text(exc)).body()
            events.put_nowait(ServerSentEvent(data=body, event="error"))
        finally:
            events.put_nowait(None)

    task = asyncio.create_task(work())
    try:
        while (event := await events.get()) is not None:
            yield event
    finally:
        task.cancel()  # the caller went away: stop paying for an answer nobody will read


@router.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse("/docs")


async def problem_response(request: Request, problem: Problem) -> JSONResponse:
    headers = {"Retry-After": str(problem.retry_after_s)} if problem.retry_after_s else None
    return JSONResponse({"detail": problem.body()}, status_code=problem.status, headers=headers)


class BodyLimit:
    """Refuses a request whose body is larger than any question, before a byte of it is read.

    The question is checked for length only after its whole body has been read into memory, so on a
    small machine one large upload could use it all. The declared length is what is checked: the
    server reads no more than that, and a body sent without one is refused."""

    def __init__(self, app: ASGIApp, max_bytes: int = MAX_BODY_BYTES) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope["method"] not in ("GET", "HEAD", "OPTIONS"):
            declared = Headers(scope=scope).get("content-length", "")
            problem = None
            if not declared.isdigit():
                problem = Problem(411, "length_required", "Send the request body with a Content-Length.")
            elif int(declared) > self.max_bytes:
                problem = Problem(413, "too_large", f"The request body is over {self.max_bytes} bytes.")
            if problem is not None:
                refusal = JSONResponse({"detail": problem.body()}, status_code=problem.status)
                await refusal(scope, receive, send)
                return
        await self.app(scope, receive, send)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Open the agent's clients once, for as long as the server runs."""
    service: Service = app.state.service
    if service.answerer is not None or not service.live:
        yield
        return
    async with demo.live(service.settings, tags=("api",)) as answerer:
        service.answerer = answerer
        try:
            yield
        finally:
            service.answerer = None
            tracing.flush()


def create_app(
    settings: Settings | None = None,
    *,
    answerer: demo.Answerer | None = None,
    quota: Quota | None = None,
    recorded: Mapping[str, demo.DemoResult] | None = None,
    live: bool | None = None,
) -> FastAPI:
    """The API. With no arguments everything comes from the environment and .env; tests pass the parts."""
    settings = settings or Settings.from_env()
    if live is None:
        live = answerer is not None or demo.can_answer(settings)
    app = FastAPI(
        title="BioScout API",
        version=__version__,
        summary="Biomedical answers that cite PubMed for every sentence and check their citations",
        description=DESCRIPTION,
        lifespan=lifespan,
    )
    app.state.service = Service(
        settings=settings,
        quota=quota or Quota.from_settings(settings),
        recorded=demo.load_recorded(settings.examples_path) if recorded is None else recorded,
        live=live,
        answerer=answerer,
    )
    app.add_exception_handler(Problem, problem_response)  # type: ignore[arg-type]
    app.add_middleware(BodyLimit)  # added before CORS, so CORS wraps it and a refusal is readable by a page
    if settings.api_cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.api_cors_origins),
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type"],
        )
    app.include_router(router)
    return app


def main() -> None:
    import uvicorn

    settings = Settings.from_env()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    uvicorn.run(create_app(settings), host=settings.api_host, port=settings.api_port)


if __name__ == "__main__":
    main()
