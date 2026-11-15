# BioScout: notes for Claude Code

Portfolio project: biomedical research agent + evaluation system. The plan of record is
`docs/design.html`; status is `docs/progress.html`.

## Commands

- `uv sync`: install
- `uv run pytest -q`: unit tests, fully offline (respx mocks + `tests/fixtures`); add `--extra demo`
  after `uv run` to include the demo page test
- `uv run ruff check . && uv run ruff format .`: lint and format (CI fails on either)
- `uv run python scripts/smoke_live.py`: real-API check of all MCP tools with the cache off (needs network)
- `uv run bioscout-mcp`: run the MCP server over stdio
- `uv run --extra demo streamlit run app/streamlit_app.py`: the demo page (needs `ANTHROPIC_API_KEY`,
  about $0.2 per question; with `BIOSCOUT_CACHE_MODE=readonly` it only replays recorded answers)
- `uv run bioscout-eval run --variants A,B,C --limit 10`: eval run (needs `ANTHROPIC_API_KEY`, costs money);
  `check-datasets` and `retrieval` are free and need only PubMed
- `uv run bioscout-eval judge evals/runs/<run_id> --variants C --limit 10`: judge a run's citations and
  add citation precision and coverage to its report (one Claude call per cited sentence, costs money)
- `uv run bioscout-eval calibrate-sample evals/runs/<run_id>` then `calibrate-score`: write 60 pairs to
  `evals/review/calibration.md` for the author to label, then score the judge against the labels
  (Cohen's kappa). Free. Never fill in or change the labels: they are the author's judgment.

## Layout

- `src/bioscout/sources/`: PubMed (E-utilities), Europe PMC, ClinicalTrials.gov clients
- `src/bioscout/http.py`, `cache.py`: shared rate-limited HTTP + SQLite response cache
- `src/bioscout/mcp_server.py`: MCP SDK 2.x `MCPServer` (FastMCP was renamed in mcp 2.x)
- `src/bioscout/answer.py`: `CitedAnswer`, the answer contract every variant, the demo and scoring share;
  its `steps` record what each step cost and how long it took
- `src/bioscout/prompts.py`: answer schema, rules and the grounded-answer prompt that B and C share
- `src/bioscout/agent/`: variants C and D, a LangGraph graph over `bioscout.llm` (plan, search, screen,
  answer; D adds verify, revise, finish)
- `src/bioscout/llm.py`: Claude calls with JSON-schema output, `fallbacks: "default"`, cost, response cache
- `src/bioscout/verifier/`: the citation judge (sentence + cited abstracts -> supported / partial /
  unsupported) and what variant D does with its verdicts (keep supporting citations, revise once)
- `src/bioscout/evals/`: datasets, variants (A to D), deterministic metrics, runner, run-level judging
  (`judgments.jsonl`, citation precision and coverage), judge calibration against hand labels,
  cost and latency tables (`costs.py`), `bioscout-eval` CLI
- `evals/datasets/`: committed JSONL; `evals/runs/` is ignored; `evals/reports/` is committed;
  `evals/review/` (abstract quotes for hand review) is ignored
- `tests/fakes.py`: the scripted LLM and in-memory PubMed that eval and agent tests share
- `src/bioscout/demo.py`: what the demo page shows for a question, as plain data (no UI code);
  `app/streamlit_app.py` only lays it out
- Coming: a Dockerfile and the public deployment of the demo

## Rules

- Eval-first: a change to agent behaviour is not done until it has a number from `evals/`.
- Every variant uses the same model, effort and answer schema; change them for all variants or none.
- B and C answer through the same prompt in `prompts.py`, so B-vs-C differences come from retrieval.
  A's and B's prompt hashes are pinned in `tests/test_agent.py`; update them only on purpose.
- C's first round also runs B's search (the question as typed) and screens its hits first, so C's
  candidates always contain B's context. Keep that when changing the search step.
- D's first four steps send the same requests as C, so they replay C's cached responses and a
  D-vs-C difference comes only from the verify step. C's prompt hash is pinned in
  `tests/test_verify.py`. Run D only with a calibrated judge: its answers depend on the judge prompt.
- Dataset PMIDs come from live PubMed results, never from memory; run `check-datasets` after edits.
- Tests never touch the network. Fixtures are synthetic and labelled as such; never put fake
  "real-looking" citations anywhere outside `tests/fixtures`.
- Do not commit article full text or abstracts. Datasets store PMIDs, questions and our own labels.
- Never commit `.env` or the cache.
- After each work session, update `docs/progress.html` (the `PROGRESS` object at the top of its
  script) and add a log entry: what was decided, what failed, which numbers moved.
- No schedule or timeline wording in code, tracked files, commit messages or branch names: no
  "Day N", planned dates, or how long something took. `docs/progress.html` shows status by phase
  only. How fast the project was built is not what it is about.
- Everything committed is English by default: code, comments, commit messages, README, reports
  and the templates of generated files. The pages in `docs/` are Korean for the author.
- HTML docs share `docs/assets/style.css`; keep its colour tokens (the verdict colours mean
  supported / partial / unsupported everywhere).
- Commit messages: imperative mood, one topic per commit.
