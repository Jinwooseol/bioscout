"""Choosing the abstracts for the answer by embedding similarity instead of a model call.

Variant C's screening step reads up to 25 candidate abstracts with the model; it is two thirds of what
the agent costs. Here the question and every candidate are embedded and the candidates closest to the
question are kept. No model reads the candidates, so nothing asks for a second search round either.

The embedder is a small local model run with fastembed (ONNX on the CPU), an optional extra (`embed`).
There is no vector database: a question has a few dozen candidates at most, and comparing the question
with all of them is exact and takes a moment.
"""

from __future__ import annotations

import asyncio
import math
import threading
from collections.abc import Sequence
from functools import cache
from typing import Any, Protocol

from bioscout.models import ArticleAbstract

EMBED_MODEL = "BAAI/bge-small-en-v1.5"  # 384 dimensions, about 70 MB; part of variant E's fingerprint


class Embedder(Protocol):
    name: str

    def embed_query(self, text: str) -> Sequence[float]: ...

    def embed_passages(self, texts: list[str]) -> list[Sequence[float]]: ...


class Ranker(Protocol):
    name: str  # what ranks, for the variant's fingerprint and the run's records

    async def rank(self, question: str, articles: list[ArticleAbstract]) -> list[tuple[str, float]]:
        """(PMID, score) for every article, best first."""
        ...


def passage_text(article: ArticleAbstract) -> str:
    """What is embedded for a paper: its title and abstract. A paper with no abstract is its title."""
    return f"{article.title}\n{article.abstract_text}".strip()


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return sum(x * y for x, y in zip(a, b, strict=True)) / norm if norm else 0.0


class EmbeddingRanker:
    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self.name = f"embedding:{embedder.name}"

    async def rank(self, question: str, articles: list[ArticleAbstract]) -> list[tuple[str, float]]:
        if not articles:
            return []
        return await asyncio.to_thread(self._rank, question, articles)  # the model runs on the CPU

    def _rank(self, question: str, articles: list[ArticleAbstract]) -> list[tuple[str, float]]:
        query = self.embedder.embed_query(question)
        vectors = self.embedder.embed_passages([passage_text(article) for article in articles])
        scored = [
            (article.pmid, cosine(query, vector)) for article, vector in zip(articles, vectors, strict=True)
        ]
        return sorted(scored, key=lambda pair: -pair[1])  # stable: equal scores keep the search order


class FastEmbedEmbedder:
    """A fastembed text model, loaded on first use (the first use also downloads it)."""

    def __init__(self, model: str = EMBED_MODEL) -> None:
        self.name = model
        self._model: Any = None
        self._lock = threading.Lock()

    def _loaded(self) -> Any:
        with self._lock:
            if self._model is None:
                try:
                    from fastembed import TextEmbedding
                except ImportError as exc:
                    raise RuntimeError(
                        "Embedding rank needs the embed extra: run `uv sync --extra embed`."
                    ) from exc
                self._model = TextEmbedding(model_name=self.name)
            return self._model

    def embed_query(self, text: str) -> Sequence[float]:
        return next(iter(self._loaded().query_embed(text))).tolist()

    def embed_passages(self, texts: list[str]) -> list[Sequence[float]]:
        return [vector.tolist() for vector in self._loaded().passage_embed(texts)]


@cache
def default_ranker(model: str = EMBED_MODEL) -> EmbeddingRanker:
    """One ranker per model for the whole process, so the model is loaded once."""
    return EmbeddingRanker(FastEmbedEmbedder(model))
