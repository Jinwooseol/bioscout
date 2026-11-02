# BioScout: notes for Claude Code

One-week portfolio project: biomedical research agent + evaluation system. The plan of record is
`docs/design.html`; daily status is `docs/progress.html`.

## Commands

- `uv sync`: install
- `uv run pytest -q`: unit tests, fully offline (respx mocks + `tests/fixtures`)
- `uv run ruff check . && uv run ruff format .`: lint and format (CI fails on either)
- `uv run python scripts/smoke_live.py`: real-API check of all MCP tools (needs network)
- `uv run bioscout-mcp`: run the MCP server over stdio

## Layout

- `src/bioscout/sources/`: PubMed (E-utilities), Europe PMC, ClinicalTrials.gov clients
- `src/bioscout/http.py`, `cache.py`: shared rate-limited HTTP + SQLite response cache
- `src/bioscout/mcp_server.py`: MCP SDK 2.x `MCPServer` (FastMCP was renamed in mcp 2.x)
- Coming: `agent/` (Day 3), `verifier/` (Day 4), `evals/` (Day 2), `app/` (Day 6)

## Rules

- Eval-first: a change to agent behaviour is not done until it has a number from `evals/`.
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
