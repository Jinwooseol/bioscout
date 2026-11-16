# BioScout

[![ci](https://github.com/Jinwooseol/bioscout/actions/workflows/ci.yml/badge.svg)](https://github.com/Jinwooseol/bioscout/actions/workflows/ci.yml)

**A biomedical research agent that cites PubMed for every claim, and checks its own citations.**
[Live page with recorded answers](https://brakey-bioscout.static.hf.space)

LLMs answering biomedical questions fail in a specific, dangerous way: they cite papers that do not
exist, or real papers that do not say what is claimed. BioScout is built eval-first around that
failure. Every sentence in an answer carries PMIDs, every PMID is checked against PubMed, and every
(claim, abstract) pair is judged for support before the answer is shown.

> Status: the MCP tool server, the eval harness and its datasets, the research agent, the citation
> judge, the verify step, the CI eval gate and the demo are built, and variants A to C have been run
> and judged on all 115 questions. Still open: checking the judge against hand labels, then running
> variant D. Progress: [`docs/progress.html`](docs/progress.html).

| Area | What was built | Where |
| --- | --- | --- |
| Agent | A LangGraph state graph: plan, search (MeSH query rewriting), screen, cited answer, verify, revise once | `src/bioscout/agent/`, `src/bioscout/verifier/` |
| Tools | An MCP server with five tools over PubMed, Europe PMC and ClinicalTrials.gov, behind rate limits, retries and a SQLite response cache; the agent can run on the same tools over MCP | `src/bioscout/mcp_server.py`, `src/bioscout/tools.py` |
| Evaluation | 115 questions, four variants on one model, deterministic metrics, an LLM judge with a hand-label calibration tool, failure classes, cost and latency per step | `src/bioscout/evals/`, `evals/` |
| Tracing | One trace per answer: steps, model calls and PubMed calls as OpenTelemetry spans, sent to Langfuse when keys are set | `src/bioscout/tracing.py` |
| CI | 124 offline tests, lint, the container build, and an eval gate on the committed report | `.github/workflows/ci.yml` |
| Demo | A Streamlit page, a Docker image, and a static page with recorded answers | `app/`, `space/` |

## Results

First full run: 100 PubMedQA questions and 15 hand-written MSI / digital PCR / liquid biopsy
questions, the same model and effort for every variant
([`evals/reports/20261116-105242-ABCE.md`](evals/reports/20261116-105242-ABCE.md); A to C were first
measured in [`20261115-195946-ABC`](evals/reports/20261115-195946-ABC.md), which also has their latency).

| Variant | Fabricated PMIDs | Sentences cited | Source paper used (PubMedQA) | Citation precision | Citation coverage | Cost / query |
| --- | --- | --- | --- | --- | --- | --- |
| A. LLM only | 0.0% | 15.7% | n/a | 51.0% | 6.3% | $0.016 |
| B. Naive RAG | 0.0% | 87.1% | 85% | 90.8% | 81.5% | $0.024 |
| C. Agent | 0.0% | 99.7% | 91% | 86.1% | 93.4% | $0.148 |
| D. Agent + verifier | built, not run yet | | | | | |
| E. Agent, embedding rank | 0.0% | 98.6% | 87% | 85.5% | 91.7% | $0.055 |

Citation precision is the share of citations whose abstract clearly backs the sentence; citation
coverage is the share of answer sentences fully backed by the abstracts they cite. Both come from an
LLM judge over all 1,151 cited sentences. They are provisional until the judge has been checked
against hand labels.

The median answer takes 8 s (A), 5 s (B) and 20 s (C). Two thirds of the agent's cost is its screening
step, which reads up to 25 candidate abstracts per search round; the report breaks cost and latency
down per variant and per step.

- No variant cited a PMID that does not exist (0 of 1,681 citations), so an existence check alone
  does not separate the variants.
- The LLM alone rarely cites, and half of what it cites from memory does not clearly back the
  sentence. 11 of its 102 citations are unsupported, several of them real papers on an unrelated
  topic; all of them pass the existence check.
- Naive RAG is precise when its search finds something, but 24 of its 115 answers have no backed
  sentence: PubMed returns nothing for 13 of the 15 long domain questions searched as typed.
- The agent backs 93% of its sentences and leaves no answer without evidence. It cites an unrelated
  paper 0.7% of the time; its misses are mostly partial support, which the verifier (variant D) is
  meant to remove.

**Where answers go wrong.** The report also classifies failures from the run's records. Accuracy looks
alike across the variants (75 to 77%), but the mistakes differ. The LLM alone answers the opposite of
the label 11 times in 100; naive RAG does so once and the agent twice. 19 of the agent's 24 wrong
answers are a "maybe" where the label says yes or no, and 19 of the 24 were written with the
question's own source paper among the abstracts: PubMedQA scores against one paper's conclusion, and
the agent reads ten. Naive RAG's wrong answers are mostly searches that found nothing (13 questions
with no answer). On the domain questions the agent loses must-cite papers at the search step (19 of
43 never returned by any search), not at selection (4).

**Choosing abstracts without the model (variant E).** Screening is two thirds of the agent's cost, and
the failure analysis says papers are lost at search, not at selection. Variant E therefore replaces
the screening call with embedding similarity: a 67 MB local model (`BAAI/bge-small-en-v1.5`), and no
vector database, since a question has at most 25 candidates. Run only up to the choice of abstracts
([`evals/reports/20261116-094814-select-CE.md`](evals/reports/20261116-094814-select-CE.md)), which costs nothing
because the plan calls replay from the cache, E keeps the source paper for 87 of 100 PubMedQA
questions against C's 91, at $0.017 a question up to that point against $0.114. The ranking itself
dropped nothing: E kept all 87 source papers its search had found. What it lacks is C's second search
round, which found five more.

Run in full and judged (A, B and C replayed from the cache at no cost, so the four variants share one
report), E costs 37% of
what C does and cites about as well: citation precision 85.5% against 86.1%, citation coverage 91.7%
against 93.4%, and no answer without a backed sentence. Accuracy is 73% against 76%; on the questions
where the source paper was among the abstracts the two are level (71 of 87, 72 of 91).

**Citation guard.** In E's first run one answer held a malformed citation: `24, 41022185` in one
citation slot, where the second number is a paper it was given. The existence check counted it as
fabricated and the eval gate refused that report ([`20261116-095853-ABCE`](evals/reports/20261116-095853-ABCE.md))
as the baseline. Answers written from abstracts may now cite only those abstracts: a garbled entry is
reduced to the given PMIDs found in it, anything else is removed, and the report counts what was
changed. The rule is part of each variant's hash, so the gate failed until B, C and E were measured
again; replayed from the cache that cost one judge call ($0.01). B and C had no citation outside their
abstracts to begin with, so their numbers did not move.

**Eval gate.** CI cannot run the agent: it has no API key, and the response cache that would replay a
run holds abstracts, which are not committed. It gates on the committed evidence instead.
`bioscout-eval gate` fails when the report named in [`evals/baseline.json`](evals/baseline.json) was
not made with the prompts, model, judge and questions in the commit, when it holds a fabricated PMID
or a failed question, or when citation precision or coverage fell more than 3 points (accuracy more
than 5) since the previous baseline. A change to the agent therefore cannot land without a new run.

## Demo

**Live page: <https://brakey-bioscout.static.hf.space>** (answers recorded from the real agent for three example questions; it
runs no model).

A one-page Streamlit app: ask a question, watch the steps run, and read the answer with a verdict
and the abstract passage behind every sentence, the cited papers, what was removed, and what it
cost.

```bash
uv run --extra demo streamlit run app/streamlit_app.py
```

A live question needs `ANTHROPIC_API_KEY` in `.env` and costs about $0.2. The three example questions
show recorded answers at once, with no key and at no cost.

As a container (port 7860, ready for a Hugging Face Docker Space):

```bash
docker build -t bioscout-demo .
docker run --rm -p 7860:7860 -e ANTHROPIC_API_KEY=... bioscout-demo
```

The live page is a static Hugging Face Space that reads the same recorded answers as the app
(`app/examples.json`). `scripts/deploy_space.py` publishes it; `--sdk docker` publishes the Streamlit
app itself instead, which Hugging Face hosts only on a paid plan.

## Architecture

```
Streamlit demo ─┐                 ┌─> report (results, costs, failure classes) ─> CI eval gate
Eval harness ───┴─> Research agent (plan → search → screen → cited answer → verify ↺ revise)
                         │ in-process, or MCP tool calls with BIOSCOUT_TOOLS=mcp
                         ▼
MCP hosts ────────> Bio MCP server ── SQLite cache ──> NCBI E-utilities, Europe PMC, ClinicalTrials.gov
(Claude Code, Claude Desktop)
```

By default the agent calls the PubMed client in-process. With `BIOSCOUT_TOOLS=mcp` it calls the
server's tools through an MCP client session instead; both end in the same client and cache, and the
example questions replay to identical answers either way.

Every answer records the cost, tokens and latency of each step. With `LANGFUSE_PUBLIC_KEY` and
`LANGFUSE_SECRET_KEY` in `.env` (and `uv sync --extra trace`), each answer is also sent as one trace:
the graph's steps, every model call with its tokens and cost, and every PubMed call, as OpenTelemetry
spans in Langfuse's attribute mapping. Tracing is off without keys, and `BIOSCOUT_TRACE_CONTENT=off`
sends names and numbers only. The export path is checked against a local receiver; it has not been
viewed in a Langfuse project yet.

Design notes, diagrams and the evaluation plan: [`docs/design.html`](docs/design.html) and
[`docs/how-it-works.html`](docs/how-it-works.html).

## The MCP server (usable on its own)

| Tool | What it does |
| --- | --- |
| `search_pubmed` | PubMed search with boolean / MeSH syntax and year filters; returns PMID, title, journal, year, first author, DOI, PMCID |
| `get_abstracts` | Abstracts with section labels, MeSH terms and publication types for up to 50 PMIDs |
| `check_pmids` | Which PMIDs exist. Missing ones are likely fabricated citations |
| `get_fulltext` | Open-access full text from Europe PMC, split into sections, within a character budget |
| `search_trials` | ClinicalTrials.gov v2 search by text, condition, intervention and status |

All responses go through a SQLite cache: it respects NCBI rate limits, makes evaluation runs
reproducible, and lets a run be replayed with no network and at no cost (`BIOSCOUT_CACHE_MODE=readonly`).

### Quickstart

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/Jinwooseol/bioscout && cd bioscout
uv sync
cp .env.example .env          # optional: add an NCBI API key
uv run pytest                 # offline unit tests
uv run python scripts/smoke_live.py   # calls every tool against the real APIs
```

Add it to **Claude Code**:

```bash
claude mcp add bioscout -- uv --directory /path/to/bioscout run bioscout-mcp
```

Or to **Claude Desktop** (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "bioscout": {
      "command": "uv",
      "args": ["--directory", "/path/to/bioscout", "run", "bioscout-mcp"]
    }
  }
}
```

A step-by-step guide with Windows paths: [`docs/guide.html`](docs/guide.html).

## Intended use

BioScout helps people find and read biomedical literature. It is not a medical device and must not
be used to make diagnostic or treatment decisions about individual patients.

## License

MIT
