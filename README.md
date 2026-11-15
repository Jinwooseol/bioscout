# BioScout

**A biomedical research agent that cites PubMed for every claim, and checks its own citations.**

LLMs answering biomedical questions fail in a specific, dangerous way: they cite papers that do not
exist, or real papers that do not say what is claimed. BioScout is built eval-first around that
failure. Every sentence in an answer carries PMIDs, every PMID is checked against PubMed, and every
(claim, abstract) pair is judged for support before the answer is shown.

> Status: **Day 4 of 7**. The MCP tool server, the eval harness, the datasets (100 PubMedQA + 15
> domain questions), the research agent (a LangGraph graph, variant C) and the citation judge are
> built, and variants A to C have been run and judged on all 115 questions. Checking the judge
> against hand labels and the verify loop (variant D) come next.
> Progress: [`docs/progress.html`](docs/progress.html).

## Results

First full run: 100 PubMedQA questions and 15 hand-written MSI / digital PCR / liquid biopsy
questions, the same model and effort for every variant
([`evals/reports/20261115-195946-ABC.md`](evals/reports/20261115-195946-ABC.md)).

| Variant | Fabricated PMIDs | Sentences cited | Source paper used (PubMedQA) | Citation precision | Citation coverage | Cost / query |
| --- | --- | --- | --- | --- | --- | --- |
| A. LLM only | 0.0% | 15.7% | n/a | 51.0% | 6.3% | $0.016 |
| B. Naive RAG | 0.0% | 87.1% | 85% | 90.8% | 81.5% | $0.024 |
| C. Agent | 0.0% | 99.7% | 91% | 86.1% | 93.4% | $0.148 |
| D. Agent + verifier | not built yet | | | | | |

Citation precision is the share of citations whose abstract clearly backs the sentence; citation
coverage is the share of answer sentences fully backed by the abstracts they cite. Both come from an
LLM judge over all 1,151 cited sentences. They are provisional until the judge has been checked
against hand labels.

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
