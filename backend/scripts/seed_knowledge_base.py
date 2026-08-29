"""One-off script to (re-)ingest the knowledge base into pgvector.

Usage: python -m scripts.seed_knowledge_base

The API also runs this automatically on startup if no KB documents
exist yet, so this script is mainly for re-ingesting after editing the
markdown content under app/rag/knowledge_base/.
"""

import asyncio

from app.db.session import AsyncSessionLocal
from app.rag.ingest import ingest_knowledge_base


async def main() -> None:
    async with AsyncSessionLocal() as db:
        count = await ingest_knowledge_base(db)
        print(f"Ingested {count} chunks into the knowledge base.")


if __name__ == "__main__":
    asyncio.run(main())
