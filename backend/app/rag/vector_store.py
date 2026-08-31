"""pgvector-backed similarity search over knowledge-base chunks.

Uses the same Postgres instance as the rest of the app (no separate
vector DB service) — the `kb_chunks.embedding` column is a pgvector
`vector(embedding_dim)` column, queried with cosine distance.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.kb_document import KBChunk, KBDocument
from app.db.models.user_document import UserDocument, UserDocumentChunk
from app.rag.embeddings import embed_text

# Prefix marking a citation as one of the user's own uploads rather than a curated
# KB doc — the frontend uses this to route a "view source" click to the documents
# endpoint instead of the knowledge-base one.
USER_DOC_SLUG_PREFIX = "user-doc:"


@dataclass
class RetrievedChunk:
    slug: str
    title: str
    content: str
    similarity: float


async def similarity_search(
    db: AsyncSession, query: str, top_k: int = 4, user_id: uuid.UUID | None = None
) -> list[RetrievedChunk]:
    """Searches the curated knowledge base and, if a user is given, that user's own
    uploaded documents — one merged, re-ranked pool rather than top_k from each
    (which would silently double the amount of context handed to the model)."""
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

    chunks: list[RetrievedChunk] = [
        RetrievedChunk(
            slug=kb_document.slug,
            title=kb_document.title,
            content=kb_chunk.content,
            similarity=float(1 - dist),
        )
        for kb_chunk, kb_document, dist in result.all()
    ]

    if user_id is not None:
        user_distance = UserDocumentChunk.embedding.cosine_distance(query_embedding)
        user_stmt = (
            select(UserDocumentChunk, UserDocument, user_distance.label("distance"))
            .join(UserDocument, UserDocumentChunk.document_id == UserDocument.id)
            .where(UserDocument.user_id == user_id)
            .order_by(user_distance)
            .limit(top_k)
        )
        user_result = await db.execute(user_stmt)
        chunks.extend(
            RetrievedChunk(
                slug=f"{USER_DOC_SLUG_PREFIX}{user_document.id}",
                title=user_document.filename,
                content=user_chunk.content,
                similarity=float(1 - dist),
            )
            for user_chunk, user_document, dist in user_result.all()
        )
        chunks.sort(key=lambda c: c.similarity, reverse=True)
        chunks = chunks[:top_k]

    return chunks
