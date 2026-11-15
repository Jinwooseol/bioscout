"""The demo: its data logic offline, and the Streamlit page driven by Streamlit's test runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from bioscout import demo
from bioscout.agent.prompts import PLAN_SCHEMA, SCREEN_SCHEMA
from bioscout.answer import CitedAnswer, CitedSentence, DroppedSentence, Step, Usage, Verdict
from bioscout.prompts import ANSWER_SCHEMA
from bioscout.verifier.judge import JUDGE_SCHEMA
from bioscout.verifier.revise import REVISE_SCHEMA
from tests.fakes import FakePubMed, RoutedLLM, reply, synthetic_abstract

TEXTS = {
    "10000001": "Drug X lowered mortality compared with drug Y in 400 adults (hazard ratio 0.70).",
    "10000002": "Drug X caused more nausea than placebo.",
}
QUOTE = "Drug X lowered mortality compared with drug Y"


def judged(sentence: str, *citations: tuple[str, str, str]) -> dict[str, Any]:
    return {
        "citations": [{"pmid": p, "verdict": v, "evidence": e} for p, v, e in citations],
        "sentence_verdict": sentence,
        "missing": "",
    }


def scripted_llm(answer: dict[str, Any], judgments: list, revisions: list | None = None) -> RoutedLLM:
    screen = {"relevant_pmids": ["10000001", "10000002"], "enough": True, "missing": "", "next_queries": []}
    return RoutedLLM(
        [],
        routes=[
            (PLAN_SCHEMA, [{"searches": [{"purpose": "all", "query": "q1"}]}]),
            (SCREEN_SCHEMA, [screen]),
            (ANSWER_SCHEMA, [answer]),
            (JUDGE_SCHEMA, judgments),
            (REVISE_SCHEMA, revisions or []),
        ],
    )


async def test_a_question_gets_a_verified_answer_its_sources_and_live_progress():
    answer = reply(
        "yes", ("X lowered mortality versus Y.", ["10000001"]), ("X is cheap.", ["10000002", "10000001"])
    )
    judgments = [
        judged("supported", ("10000001", "supported", QUOTE)),
        judged("unsupported", ("10000002", "unsupported", ""), ("10000001", "unsupported", "")),
    ]
    revisions = [{"sentences": [{"index": 1, "text": "", "citations": []}]}]
    seen: list[str] = []
    result = await demo.answer_question(
        scripted_llm(answer, judgments, revisions),
        FakePubMed(texts=TEXTS),
        "Is X better than Y?",
        on_step=lambda step: seen.append(step.name),
    )

    assert seen == ["plan", "search", "screen", "answer", "verify", "revise", "verify"]
    assert [(s.text, s.verdict) for s in result.answer.sentences] == [
        ("X lowered mortality versus Y.", Verdict.SUPPORTED)
    ]
    assert [d.reason for d in result.answer.dropped] == ["not_backed"]
    # The cards come from the abstracts the agent fetched: only what the answer still cites.
    assert set(result.articles) == {"10000001"}
    (source,) = demo.sources_of(result)
    assert (source.pmid, source.sentences, source.article.title) == ("10000001", [1], "Synthetic 10000001")
    assert source.url == "https://pubmed.ncbi.nlm.nih.gov/10000001/"


async def test_the_check_can_be_turned_off():
    llm = scripted_llm(reply("yes", ("X lowered mortality versus Y.", ["10000001"])), judgments=[])
    result = await demo.answer_question(llm, FakePubMed(texts=TEXTS), "Is X better than Y?", verify=False)
    assert [s.verdict for s in result.answer.sentences] == [None]
    assert [s.name for s in result.answer.steps] == ["plan", "search", "screen", "answer"]


def canned_result() -> demo.DemoResult:
    usage = Usage(usd=0.05, latency_ms=1500, llm_calls=1)
    answer = CitedAnswer(
        question="Is X better than Y?",
        decision="yes",
        sentences=[
            CitedSentence(
                text="X lowered mortality versus Y.",
                citations=["10000001", "10000002"],
                verdict=Verdict.SUPPORTED,
                evidence=QUOTE,
            ),
            CitedSentence(text="X also caused nausea.", citations=["10000002"], verdict=Verdict.PARTIAL),
        ],
        dropped=[DroppedSentence(text="X is cheap.", citations=["99999999"], reason="unsupported")],
        steps=[
            Step(name="search", usage=Usage(latency_ms=400, tool_calls=3)),
            Step(name="verify", usage=usage),
            Step(name="verify", usage=usage),
        ],
    )
    articles = {"10000001": synthetic_abstract("10000001", "Drug X lowered mortality.")}
    return demo.DemoResult(answer=answer, articles=articles)


def test_sources_follow_the_order_of_first_citation_and_steps_are_added_up():
    result = canned_result()
    sources = demo.sources_of(result)
    assert [(s.pmid, s.sentences) for s in sources] == [("10000001", [1]), ("10000002", [1, 2])]
    assert sources[1].article is None  # the agent never fetched it, and the page says so

    rows = {row["Step"]: row for row in demo.step_rows(result.answer)}
    verify = rows[demo.STEP_LABELS["verify"]]
    assert (verify["Runs"], verify["LLM calls"], verify["Cost (USD)"], verify["Seconds"]) == (2, 2, 0.1, 3.0)
    assert demo.totals(result.answer) == (pytest.approx(0.10), pytest.approx(3.4))
    assert not demo.was_replayed(result.answer)
    cached = Step(name="answer", usage=Usage(usd=0.05, llm_calls=2, cached_llm_calls=2))
    assert demo.was_replayed(CitedAnswer(question="Q?", steps=[cached]))


def test_the_page_shows_the_answer_with_verdicts_sources_and_what_was_removed(monkeypatch):
    testing = pytest.importorskip("streamlit.testing.v1", reason="the demo extra is not installed")

    async def fake_ask(question: str, verify: bool = True, on_step=None) -> demo.DemoResult:
        if on_step:
            on_step(Step(name="plan", usage=Usage(latency_ms=1200)))
        return canned_result()

    monkeypatch.setattr(demo, "ask", fake_ask)
    monkeypatch.setattr(demo, "can_answer", lambda: True)
    page = Path(__file__).parent.parent / "app" / "streamlit_app.py"
    app = testing.AppTest.from_file(str(page), default_timeout=30).run()
    assert not app.exception and app.title[0].value == "BioScout"

    app.text_area[0].set_value("Is X better than Y?").run()  # typing enables the Ask button
    next(b for b in app.button if b.label == "Ask").click().run()
    assert not app.exception

    shown = "\n".join(m.value for m in app.markdown)
    assert "**1.** X lowered mortality versus Y." in shown and ":green-badge[Supported]" in shown
    assert ":orange-badge[Partly supported]" in shown and "~~X is cheap.~~" in shown
    assert "[PMID 10000001](https://pubmed.ncbi.nlm.nih.gov/10000001/)" in shown
    assert "**[Synthetic 10000001]" in shown and "**[Not among the abstracts the agent read]" in shown
    captions = "\n".join(c.value for c in app.caption)
    assert f"“{QUOTE}”" in captions and "Still not backed after one rewrite." in captions
    assert "Questions left in this session: 4" in [c.value for c in app.sidebar.caption]
