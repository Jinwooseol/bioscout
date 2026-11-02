# BioScout

**A biomedical research agent that cites PubMed for every claim, and checks its own citations.**

LLMs answering biomedical questions fail in a specific, dangerous way: they cite papers that do not
exist, or real papers that do not say what is claimed. BioScout is built eval-first around that
failure. Every sentence in an answer carries PMIDs, every PMID is checked against PubMed, and every
(claim, abstract) pair is judged for support before the answer is shown.

> Status: **Day 1 of 7**. The MCP tool server is done and tested. Agent, verifier and evaluation
> results land on Days 2 to 5. Progress: [`docs/progress.html`](docs/progress.html).

## Results

Pending. The evaluation compares four variants on 100 PubMedQA questions and 25 hand-written
MSI / digital PCR / liquid biopsy questions:

| Variant | Fabricated PMIDs | Citation precision | Citation coverage | Cost / query |
| --- | --- | --- | --- | --- |
| A. LLM only | | | | |
| B. Naive RAG | | | | |
| C. Agent | | | | |
| D. Agent + verifier | | | | |

## Architecture

```
Streamlit demo ─┐                 ┌─> results table, CI gate
Eval harness ───┴─> Research agent (plan → search → cited answer → verify ↺)
                         │ MCP tool calls                       │ traces
                         ▼                                      ▼
                    Bio MCP server ── SQLite cache ──> NCBI E-utilities, Europe PMC, ClinicalTrials.gov
                                                                  Langfuse
```

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
reproducible, and lets CI replay recorded responses with no network (`BIOSCOUT_CACHE_MODE=readonly`).

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
