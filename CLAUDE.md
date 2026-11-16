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
- `uv run --extra demo --extra trace python scripts/trace_check.py`: replay the first example question
  with tracing on (free, read-only cache). Needs `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` in
  `.env`; with them set, the demo, `bioscout-eval run` and `judge` send traces too
- `BIOSCOUT_TOOLS=mcp` before a demo or eval run: the agent calls the MCP server's tools over a client
  session instead of the PubMed client in-process. Same answers; the run's metadata records which
- `uv run --extra api bioscout-api`: the HTTP API on http://127.0.0.1:8000 (docs at `/docs`). With
  `BIOSCOUT_CACHE_MODE=readonly` it only replays; `BIOSCOUT_QUOTA_PATH` moves the count of questions
- `uv run python scripts/build_examples.py`: record the demo's example answers into `app/examples.json`
  (free for questions already in the cache)
- `docker build -t bioscout-demo .`: the image of the demo page and the API. There is no Docker on the
  author's PC; the CI `image` job builds it and checks that the page and the API start
- `fly deploy --ha=false`: the API on Fly.io from `fly.toml` (README, "Hosting the API"). Not run yet:
  it needs the author's Fly account, and the keys set there with `fly secrets set`
- `uv run --with huggingface_hub python scripts/deploy_space.py brakey/bioscout`: publish the public page
  (a static Space with the recorded answers). `--dry-run` lists the files; `--sdk docker` publishes the
  Streamlit app instead, which Hugging Face refuses without a paid plan (HTTP 402)
- `uv run bioscout-eval run --variants A,B,C --limit 10`: eval run (needs `ANTHROPIC_API_KEY`, costs money);
  `check-datasets` and `retrieval` are free and need only PubMed
- `uv run bioscout-eval judge evals/runs/<run_id> --variants C --limit 10`: judge a run's citations and
  add citation precision and coverage to its report (one Claude call per cited sentence, costs money)
- `uv run bioscout-eval calibrate-sample evals/runs/<run_id>` then `calibrate-score`: write 60 pairs to
  `evals/review/calibration.md` for the author to label, then score the judge against the labels
  (Cohen's kappa). Free. Never fill in or change the labels: they are the author's judgment.
- `BIOSCOUT_CACHE_MODE=readonly uv run --extra embed bioscout-eval selection --variants C,E --concurrency 1`:
  run C and E up to the choice of abstracts and compare which expected papers were chosen. No answer
  is written; free when the cache is read-only. `bioscout-eval run --variants E` runs E in full (paid)
- `uv run bioscout-eval gate --since HEAD`: the eval gate, offline and free. It checks that the report
  named in `evals/baseline.json` was made with the prompts, model, judge and questions in the code,
  holds no fabricated PMID, and did not drop beyond the limits since the baseline at that revision.
  CI runs it as the `eval-gate` job

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
- `src/bioscout/tools.py`: `McpLiterature`, the agent's PubMed interface over an MCP client session
- `src/bioscout/evals/failures.py`: failure classes from a run's records (where an expected paper was
  lost, kinds of wrong answer); the report's last section
- `src/bioscout/agent/rank.py`: variant E's choice of abstracts by embedding similarity (`Ranker`,
  fastembed model, optional extra `embed`); `src/bioscout/evals/selection.py`: the selection-only run
- `src/bioscout/tracing.py`: OpenTelemetry spans in Langfuse's attribute mapping (optional extra
  `trace`); off, and wrapping nothing, unless keys or an OTLP endpoint are configured
- `docs/jd-fit.html`: what job postings ask for against what the project does, with the gaps
- `src/bioscout/evals/gate.py`, `evals/baseline.json`: the eval gate and the report it holds the code to
- `evals/datasets/`: committed JSONL; `evals/runs/` is ignored; `evals/reports/` is committed;
  `evals/review/` (abstract quotes for hand review) is ignored
- `tests/fakes.py`: the scripted LLM and in-memory PubMed that eval and agent tests share
- `src/bioscout/demo.py`: what the demo page and the API show for a question, as plain data (no UI or
  web code); `app/streamlit_app.py` and `api.py` only lay it out
- `src/bioscout/api.py`: the FastAPI service (optional extra `api`): answers as JSON or server-sent
  events, and the checks a question passes before the agent runs
- `src/bioscout/quota.py`: the cap on questions per UTC day and in total, in a SQLite file that the
  API and the demo page share
- `Dockerfile`, `.dockerignore`: the page and the API as one image (uid 1000, port 7860, no secrets in
  the image); `bioscout-api` as the command, or `BIOSCOUT_SERVE=api`, runs the API instead of the page
- `src/bioscout/container.py`: the image's entrypoint (`bioscout-serve`). It picks what to serve and,
  started as root, hands the state folders to uid 1000 and gives up root
- `fly.toml`: the API on Fly.io (one machine that stops when idle, a volume for the count of questions)
- `space/static/index.html`: the public page (https://brakey-bioscout.static.hf.space). It reads `examples.json` and
  shows the recorded answers; its labels must match `demo.py` (a test checks)
- `scripts/deploy_space.py`, `space/*/README.md`: publish to a Hugging Face Space from an allow-list of
  files, with the login saved on the PC (no token in the repository)

## Rules

- Eval-first: a change to agent behaviour is not done until it has a number from `evals/`.
- After a full run of changed agent code, point `evals/baseline.json` at the new report so the gate
  passes. Never edit a report by hand. A drop accepted on purpose goes under `accepted` with its reason.
- Every variant uses the same model, effort and answer schema; change them for all variants or none.
- B and C answer through the same prompt in `prompts.py`, so B-vs-C differences come from retrieval.
  A's and B's prompt hashes are pinned in `tests/test_agent.py`; update them only on purpose (B's and
  C's last changed when the citation guard went in).
- C's first round also runs B's search (the question as typed) and screens its hits first, so C's
  candidates always contain B's context. Keep that when changing the search step.
- D's first four steps send the same requests as C, so they replay C's cached responses and a
  D-vs-C difference comes only from the verify step. C's prompt hash is pinned in
  `tests/test_verify.py`. Run D only with a calibrated judge: its answers depend on the judge prompt.
- Run a new variant together with the measured ones (`bioscout-eval run --variants A,B,C,E`): cached
  answers and judgments replay for free and one report holds every variant. Check first, with the
  cache read-only, that the old variants replay without an error.
- The baseline is `20261116-105242-ABCE` (A, B, C, E, measured with the citation guard). The earlier
  `20261116-095853-ABCE` report is kept as the run the gate refused: one E answer had a malformed
  citation (`24, 41022185`), counted as fabricated.
- An answer written from abstracts may cite only those abstracts (`ground_citations` in `prompts.py`,
  applied in `answer_from_result`). What it changes is kept in `CitedAnswer.ungrounded` and counted in
  the report. The rule (`GROUNDING`) is in the hash of B, C, D and E: change the string when the rule
  changes. Variant A answers from memory and is left alone.
- Nothing printed, logged or stored in a record may carry the NCBI API key or contact address: build
  such text from `cache.visible_params`, never from the raw request parameters.
- Do not add fields to `AgentConfig`: it is part of C's and D's prompt hash, so the gate fails until
  they are measured again. Pass new parts (like the ranker) separately and give the new variant its
  own fingerprint. The embedding model's name is part of E's.
- Tracing must stay a no-op without configuration: `traced_llm` and `traced_literature` return the
  object they were given. Never log or print the Langfuse keys or the Authorization header. Abstract
  text stays out of the PubMed spans.
- Run the tests with `--extra demo --extra trace --extra api` so the page, tracing and API tests are
  not skipped.
- The agent knows PubMed only through `Literature`. Keep both tool paths returning the same data, so
  cached responses replay on either.
- Do not tune the agent to say yes or no where the abstracts disagree just to raise PubMedQA accuracy:
  the label is one paper's conclusion. Report hedges and opposite answers separately.
- Dataset PMIDs come from live PubMed results, never from memory; run `check-datasets` after edits.
- Tests never touch the network. Fixtures are synthetic and labelled as such; never put fake
  "real-looking" citations anywhere outside `tests/fixtures`.
- Do not commit article full text or abstracts. Datasets store PMIDs, questions and our own labels.
- Never commit `.env` or the cache.
- `app/examples.json` holds answers and bibliographic lines only: no abstract text and no passage
  quoted from one. `demo.record()` strips them and a test checks the committed file.
- The question allowance (`quota.py`) is what bounds the bill of a deployment that holds a Claude API
  key. Every path that runs the agent for a visitor takes from it first (`admit` in `api.py`,
  `request_answer` in the page): never add a path around it, never give a question back when its
  answer fails, and never let zero or a missing value mean "no limit". The public page still runs no
  model: the API is not hosted anywhere.
- The limit per caller (`Quota.take(caller)`, the API only) is a courtesy that keeps one visitor from
  using up the day. Never count on it to bound spending: a caller who changes address gets a new
  count. The quota file must never hold an address, only the salted hash, and only for the day.
  Believe a header for the caller's address only when the operator named it
  (`BIOSCOUT_API_CLIENT_HEADER`): a caller can send any header.
- A hosted API runs as one instance with `BIOSCOUT_CACHE_DIR` on a disk that outlives the container:
  the count of questions is a file. Never put a key in `fly.toml` or any tracked file; keys are set
  on the host (`fly secrets set`).
- The server never runs as root. The image starts as root only so `container.py` can hand a mounted
  folder to uid 1000; keep `give_up_root` the last thing before the command is run.
- What a visitor is told about a failed answer is `demo.failure_text`: the kind of error, never its
  text. The API logs the kind and the stack frames only. An exception can quote a request, and a
  request to PubMed carries the API key and the contact address.
- In `api.py` the checks that refuse a question live in the `admit` dependency, not in the route: an
  error raised inside a streaming route after it has started cannot become an error response.
- No test may count against the real allowance: `tests/conftest.py` points `BIOSCOUT_QUOTA_PATH` at a
  temp folder for every test. A page test also replaces `demo.quota`, so `.env` is not read.
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
