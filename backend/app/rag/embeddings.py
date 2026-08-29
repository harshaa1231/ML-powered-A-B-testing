"""Local, free embedding model wrapper (no external API calls).

Uses fastembed (ONNX runtime) rather than sentence-transformers (full
PyTorch) for the same model — same 384-dim vectors, roughly half the
peak memory, which matters when deploying to a memory-constrained
free-tier instance (e.g. Render's 512MB free web service).
"""

from __future__ import annotations

from functools import lru_cache

from fastembed import TextEmbedding

from app.core.config import get_settings

settings = get_settings()


@lru_cache
def get_embedding_model() -> TextEmbedding:
    return TextEmbedding(model_name=settings.embedding_model)


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_embedding_model()
    return [v.tolist() for v in model.embed(texts)]


def embed_text(text: str) -> list[float]:
    return embed_texts([text])[0]
