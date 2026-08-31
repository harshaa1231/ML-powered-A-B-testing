"""Shared text-chunking logic — used for both the curated knowledge base
(app/rag/ingest.py) and user-uploaded documents (app/services/document_processing.py),
so the two ingestion paths can't drift into different chunking behavior."""

from __future__ import annotations

CHUNK_SIZE = 800  # characters, roughly a couple of paragraphs
CHUNK_OVERLAP = 100


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
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
