# Eval datasets

One JSON object per line. Datasets store PMIDs, questions and labels only: no abstract or full text.
Check that every PMID still exists with `uv run bioscout-eval check-datasets`.

## `pubmedqa_100.jsonl`

100 questions sampled from PubMedQA's expert-labelled set (PQA-L, 1,000 questions).

- Source: [`data/ori_pqal.json`](https://github.com/pubmedqa/pubmedqa) (MIT licence), SHA-256
  `8b3276be8942ebbd77f3ddcda12c1749bf0e490045a736fd8438ee40cf37a41d` as downloaded on 2026-11-04.
- Built by `uv run python scripts/build_pubmedqa.py` (seed 2026). Kept fields: PMID, question,
  `final_decision`. Dropped: `CONTEXTS` (abstract text) and `LONG_ANSWER` (the abstract's conclusion).
- Labels: yes 66, no 27, maybe 7. Always answering "yes" scores 66%, so accuracy is read against that.
- Caveat: PubMedQA questions are written from the source paper's title, so searching the question
  verbatim finds the source paper unusually often (85/100 in the top 10). Results are always reported
  next to the hand-written domain set.

Fields: `id` (`pqa-<pmid>`), `dataset`, `question`, `answer` (yes / no / maybe), `pmid` (source paper).

## `domain.jsonl`

Hand-checked questions on microsatellite instability (`msi`), digital PCR (`dpcr`) and liquid biopsy
(`liquid_biopsy`): 15 now, 25 planned.

Fields: `id`, `topic`, `question`, `must_cite` (PMIDs a good answer cites), `key_points` (claims a good
answer contains, each with the must-cite PMIDs that support it), `status` (`draft` until the author has
reviewed the item against the abstracts, then `reviewed`).

Every PMID was taken from a live PubMed search and its abstract was read before use; key points are
paraphrased, never copied. Supporting abstract quotes for review live in `evals/review/`, which is
git-ignored because it quotes abstracts.
