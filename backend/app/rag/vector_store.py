"""pgvector-backed similarity search over knowledge-base chunks.

Uses the same Postgres instance as the rest of the app (no separate
vector DB service) — the `kb_chunks.embedding` column is a pgvector
`vector(embedding_dim)` column, queried with cosine distance.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.kb_document import KBChunk, KBDocument
from app.rag.embeddings import embed_text


@dataclass
class RetrievedChunk:
    slug: str
    title: str
    content: str
    similarity: float


async def similarity_search(db: AsyncSession, query: str, top_k: int = 4) -> list[RetrievedChunk]:
    query_embedding = embed_text(query)

    # cosine_distance: 0 = identical, 2 = opposite. similarity = 1 - distance.
    distance = KBChunk.embedding.cosine_distance(query_embedding)
    stmt = (
        select(KBChunk, KBDocument, distance.label("distance"))
        .join(KBDocument, KBChunk.document_id == KBDocument.id)
        .order_by(distance)
        .limit(top_k)
    )
    result = await db.execute(stmt)

    chunks: list[RetrievedChunk] = []
    for kb_chunk, kb_document, dist in result.all():
        chunks.append(
            RetrievedChunk(
                slug=kb_document.slug,
                title=kb_document.title,
                content=kb_chunk.content,
                similarity=float(1 - dist),
            )
        )
    return chunks
