# BioScout: notes for Claude Code

One-week portfolio project: biomedical research agent + evaluation system. The plan of record is
`docs/design.html`; daily status is `docs/progress.html`.

## Commands

- `uv sync`: install
- `uv run pytest -q`: unit tests, fully offline (respx mocks + `tests/fixtures`)
- `uv run ruff check . && uv run ruff format .`: lint and format (CI fails on either)
- `uv run python scripts/smoke_live.py`: real-API check of all MCP tools (needs network)
- `uv run bioscout-mcp`: run the MCP server over stdio
- `uv run bioscout-eval run --variants A,B,C --limit 10`: eval run (needs `ANTHROPIC_API_KEY`, costs money);
  `check-datasets` and `retrieval` are free and need only PubMed

## Layout

- `src/bioscout/sources/`: PubMed (E-utilities), Europe PMC, ClinicalTrials.gov clients
- `src/bioscout/http.py`, `cache.py`: shared rate-limited HTTP + SQLite response cache
- `src/bioscout/mcp_server.py`: MCP SDK 2.x `MCPServer` (FastMCP was renamed in mcp 2.x)
- `src/bioscout/answer.py`: `CitedAnswer`, the answer contract every variant, the demo and scoring share
- `src/bioscout/prompts.py`: answer schema, rules and the grounded-answer prompt that B and C share
- `src/bioscout/agent/`: variant C, a LangGraph graph (plan, search, screen, answer) over `bioscout.llm`
- `src/bioscout/llm.py`: Claude calls with JSON-schema output, `fallbacks: "default"`, cost, response cache
- `src/bioscout/evals/`: datasets, variants (A, B, C), deterministic metrics, runner, `bioscout-eval` CLI
- `evals/datasets/`: committed JSONL; `evals/runs/` is ignored; `evals/reports/` is committed;
  `evals/review/` (abstract quotes for hand review) is ignored
- `tests/fakes.py`: the scripted LLM and in-memory PubMed that eval and agent tests share
- Coming: `verifier/` (Day 4), `app/` (Day 6)

## Rules

- Eval-first: a change to agent behaviour is not done until it has a number from `evals/`.
- Every variant uses the same model, effort and answer schema; change them for all variants or none.
- B and C answer through the same prompt in `prompts.py`, so B-vs-C differences come from retrieval.
  A's and B's prompt hashes are pinned in `tests/test_agent.py`; update them only on purpose.
- Dataset PMIDs come from live PubMed results, never from memory; run `check-datasets` after edits.
- Tests never touch the network. Fixtures are synthetic and labelled as such; never put fake
  "real-looking" citations anywhere outside `tests/fixtures`.
- Do not commit article full text or abstracts. Datasets store PMIDs, questions and our own labels.
- Never commit `.env` or the cache.
- After each work session, update `docs/progress.html` (the `PROGRESS` object at the top of its
  script) and add a build-log entry: what was decided, what failed, which numbers moved.
- Docs in `docs/` are Korean for the author; README, code and comments are English for the public.
- HTML docs share `docs/assets/style.css`; keep its colour tokens (the verdict colours mean
  supported / partial / unsupported everywhere).
- Commit messages: imperative mood, one topic per commit.
