"""Chunk and embed the knowledge-base markdown docs into pgvector.

Idempotent: re-running clears and re-inserts each document's chunks, so
it's safe to call on every deploy/startup after editing the KB content.
"""

from __future__ import annotations

import re
from pathlib import Path

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.kb_document import KBChunk, KBDocument
from app.rag.embeddings import embed_texts

KB_DIR = Path(__file__).parent / "knowledge_base"
CHUNK_SIZE = 800  # characters, roughly a couple of paragraphs
CHUNK_OVERLAP = 100


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= chunk_size:
            current = f"{current}\n\n{para}" if current else para
        else:
            if current:
                chunks.append(current)
            current = para

    if current:
        chunks.append(current)

    if overlap and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = chunks[i - 1][-overlap:]
            overlapped.append(f"{tail}\n\n{chunks[i]}")
        return overlapped

    return chunks


def _parse_title(markdown: str, fallback: str) -> str:
    match = re.match(r"^#\s+(.+)$", markdown.strip(), flags=re.MULTILINE)
    return match.group(1).strip() if match else fallback


async def ingest_knowledge_base(db: AsyncSession) -> int:
    """Ingest every .md file under knowledge_base/. Returns the number of chunks written."""
    total_chunks = 0

    for path in sorted(KB_DIR.glob("*.md")):
        slug = path.stem
        content = path.read_text(encoding="utf-8")
        title = _parse_title(content, fallback=slug)

        existing = (await db.execute(select(KBDocument).where(KBDocument.slug == slug))).scalar_one_or_none()
        if existing is not None:
            await db.execute(delete(KBChunk).where(KBChunk.document_id == existing.id))
            existing.title = title
            existing.content = content
            document = existing
        else:
            document = KBDocument(slug=slug, title=title, content=content)
            db.add(document)
            await db.flush()

        chunk_texts = _chunk_text(content)
        embeddings = embed_texts(chunk_texts)

        for idx, (chunk_text, embedding) in enumerate(zip(chunk_texts, embeddings, strict=False)):
            db.add(KBChunk(document_id=document.id, chunk_index=idx, content=chunk_text, embedding=embedding))
            total_chunks += 1

    await db.commit()
    return total_chunks
