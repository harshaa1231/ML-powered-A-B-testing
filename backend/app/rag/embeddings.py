"""Local, free embedding model wrapper (no external API calls)."""

from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.core.config import get_settings

settings = get_settings()


@lru_cache
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(settings.embedding_model)


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_embedding_model()
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [v.tolist() for v in vectors]


def embed_text(text: str) -> list[float]:
    return embed_texts([text])[0]
