"""BioScout demo page: ask a biomedical research question and read an answer in which every sentence
cites PubMed and carries the verdict of the citation check.

    uv run --extra demo streamlit run app/streamlit_app.py

The page only lays out what bioscout.demo returns.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import streamlit as st

from bioscout import demo
from bioscout.answer import CitedSentence, Step, Verdict
from bioscout.http import CacheMissError

# Answers recorded for the example questions (scripts/build_examples.py), shown without calling Claude.
RECORDED = demo.load_recorded(Path(__file__).parent / "examples.json")
MAX_QUESTIONS = int(os.environ.get("BIOSCOUT_DEMO_MAX_QUESTIONS", "5"))  # per browser session
MAX_CHARS = 400
BADGES = {Verdict.SUPPORTED: "green", Verdict.PARTIAL: "orange", Verdict.UNSUPPORTED: "red", None: "gray"}
LIVE_OFF = (
    "Live questions are switched off here: no Claude API key is configured. The example questions above "
    f"show recorded answers. To ask your own, run BioScout with your own key: {demo.REPO_URL}"
)
DISCLAIMER = (
    "BioScout helps you find and read biomedical literature. It is not a medical device and must not be "
    "used to make diagnostic or treatment decisions about individual patients."
)


def badge(verdict: Verdict | None) -> str:
    return f":{BADGES[verdict]}-badge[{demo.VERDICT_LABELS[verdict]}]"


def show_sentence(number: int, sentence: CitedSentence) -> None:
    links = " ".join(f"[PMID {p}]({demo.pubmed_url(p)})" for p in sentence.citations)
    st.markdown(f"**{number}.** {sentence.text} {links}  \n{badge(sentence.verdict)}")
    if sentence.evidence:
        st.caption(f"Evidence in the abstract: “{sentence.evidence}”")


def show_result(result: demo.DemoResult, live: bool) -> None:
    answer = result.answer
    if answer.refused:
        st.warning("The model declined to answer this question.")
        return

    st.subheader("Answer")
    if result.recorded:
        st.caption(
            f"A recorded answer to an example question (it cost ${demo.totals(answer)[0]:.2f} to produce). "
            + ("Ask your own question" if live else "Run BioScout yourself")
            + " to see the agent work and the abstract passage behind each verdict."
        )
    if answer.decision in ("yes", "no", "maybe"):
        st.markdown(f"**In short: {answer.decision}.**")
    if not answer.sentences:
        st.info("Nothing in the abstracts that were found backs an answer, so no statement is shown.")
    for number, sentence in enumerate(answer.sentences, start=1):
        show_sentence(number, sentence)

    if answer.dropped:
        with st.expander(f"Removed from the answer: {len(answer.dropped)} sentence(s) that nothing backed"):
            for dropped in answer.dropped:
                st.markdown(f"~~{dropped.text}~~")
                st.caption(demo.DROP_REASONS.get(dropped.reason, dropped.reason))

    sources = demo.sources_of(result)
    if sources:
        st.subheader("Sources")
    for source in sources:
        with st.container(border=True):
            article = source.article
            title = article.title if article else "Not among the abstracts the agent read"
            st.markdown(f"**[{title}]({source.url})**")
            where = ", ".join(str(n) for n in source.sentences)
            details = [f"{article.journal} {article.year or ''}".strip()] if article else []
            st.caption(" · ".join([*details, f"PMID {source.pmid}", f"cited by sentence {where}"]))

    with st.expander("How this answer was made"):
        usd, seconds = demo.totals(answer)
        cost, time_taken, searches, read = st.columns(4)
        replayed = result.recorded or demo.was_replayed(answer)
        cost.metric("Cost", f"${usd:.2f}")
        time_taken.metric("Time", "n/a" if replayed else f"{seconds:.0f} s")
        searches.metric("PubMed searches", len(answer.searches))
        read.metric("Abstracts used", len(answer.context_pmids))
        if replayed:
            st.caption(
                "This answer was replayed from recorded responses: the cost is what it cost when "
                "first produced, and the timings are not real."
            )
        st.dataframe(demo.step_rows(answer), hide_index=True)
        st.markdown("**PubMed searches**")
        st.dataframe([{"Query": s.query, "Results": s.total_found} for s in answer.searches], hide_index=True)


def ask(question: str, verify: bool) -> None:
    """Run the agent for the question and keep the result in the session."""
    with st.status("Working on it…", expanded=True) as status:

        def on_step(step: Step) -> None:
            label = demo.STEP_LABELS.get(step.name, step.name)
            status.write(f"{label} · {step.usage.latency_ms / 1000:.1f} s")

        try:
            st.session_state.result = asyncio.run(demo.ask(question, verify=verify, on_step=on_step))
            status.update(label="Done", state="complete", expanded=False)
        except CacheMissError:
            st.session_state.result = None
            status.update(label="This question is not in the recorded answers", state="error")
            st.error("The page is replaying recorded answers only, and this question was not recorded.")
        except Exception as exc:  # a failed call must not leave the page half drawn
            st.session_state.result = None
            status.update(label="Something went wrong", state="error")
            st.error(f"The answer could not be produced: {type(exc).__name__}: {exc}")


def use_example(text: str) -> None:
    """An example click fills in the question and, when its answer is recorded, shows it at once."""
    st.session_state.question = text
    if text in RECORDED:
        st.session_state.result = RECORDED[text]


def request_answer() -> None:
    """The Ask click. It runs before the page is redrawn, so the counter in the sidebar is already right."""
    question = st.session_state.question.strip()
    if question in RECORDED:  # nothing to run, and it does not count against the session's questions
        st.session_state.result = RECORDED[question]
        return
    st.session_state.pending = question
    st.session_state.asked += 1


st.set_page_config(page_title="BioScout", layout="centered")
st.session_state.setdefault("question", "")
st.session_state.setdefault("result", None)
st.session_state.setdefault("asked", 0)
st.session_state.setdefault("pending", "")

st.title("BioScout")
st.markdown(
    "Ask a biomedical research question. BioScout searches PubMed, answers with a citation on every "
    "sentence, and checks each citation against the paper's abstract before showing it."
)

ready = demo.can_answer()
with st.sidebar:
    verify = st.toggle(
        "Check every citation",
        value=True,
        disabled=not ready,
        help="Judge each sentence against the abstracts it cites, rewrite what is not fully backed once, "
        "and remove what nothing backs. Turn off to see the answer as first written.",
    )
    left = max(0, MAX_QUESTIONS - st.session_state.asked)
    st.caption(f"Questions left in this session: {left}" if ready else "Live questions are off on this page.")
    st.caption(DISCLAIMER)

st.caption("Try an example:")
for example in demo.EXAMPLE_QUESTIONS:
    st.button(example, on_click=use_example, args=(example,), use_container_width=True)

question = st.text_area("Your question", key="question", max_chars=MAX_CHARS, height=110)
if not ready:
    st.info(LIVE_OFF)
can_ask = bool(question.strip()) and (question.strip() in RECORDED or (ready and left > 0))
st.button("Ask", type="primary", disabled=not can_ask, on_click=request_answer)
if st.session_state.pending:
    asked, st.session_state.pending = st.session_state.pending, ""
    ask(asked, verify)

if st.session_state.result is not None:
    show_result(st.session_state.result, live=ready)

st.divider()
st.caption(DISCLAIMER)
st.caption(f"[Source code and evaluation results]({demo.REPO_URL})")
