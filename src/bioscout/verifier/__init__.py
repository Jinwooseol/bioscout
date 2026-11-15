"""Citation verification: the judge, and the verify-and-revise loop that variant D runs on an answer."""

from bioscout.verifier.judge import (
    JUDGE_SHA,
    CitationJudgment,
    JudgeRefusedError,
    SentenceJudgment,
    judge_sentence,
)
from bioscout.verifier.revise import (
    VerifyConfig,
    apply_judgment,
    revise_sentences,
    unbacked_note,
    verify_fingerprint,
)

__all__ = [
    "JUDGE_SHA",
    "CitationJudgment",
    "JudgeRefusedError",
    "SentenceJudgment",
    "VerifyConfig",
    "apply_judgment",
    "judge_sentence",
    "revise_sentences",
    "unbacked_note",
    "verify_fingerprint",
]
