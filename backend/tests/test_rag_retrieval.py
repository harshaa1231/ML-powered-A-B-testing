"""Direct tests of the retrieval half of RAG — independent of which route calls it
(chat, experiment summaries, practice feedback, analytics all share this same
pipeline), so testing it once here covers the mechanism itself."""

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import get_settings
from app.rag.vector_store import similarity_search
from tests.conftest import requires_db

settings = get_settings()


async def _get_session():
    # A fresh engine per test (NullPool), same reasoning as the `client` fixture in
    # conftest.py: pytest-asyncio gives each test its own event loop, and reusing the
    # app's module-level engine across tests' loops raises "attached to a different loop".
    engine = create_async_engine(settings.database_url, poolclass=NullPool)
    session_maker = async_sessionmaker(engine, expire_on_commit=False)
    return engine, session_maker()


@requires_db
async def test_similarity_search_retrieves_relevant_kb_doc(client) -> None:
    # `client` fixture isn't used directly but pulls in _prepare_schema + a reachable DB.
    engine, db = await _get_session()
    try:
        results = await similarity_search(db, "What does a p-value actually mean?", top_k=3)
    finally:
        await db.close()
        await engine.dispose()

    assert len(results) > 0
    assert any("p-value" in r.title.lower() or "p‑value" in r.title.lower() for r in results)
    assert all(0 <= r.similarity <= 1.01 for r in results)  # cosine similarity, small float slack


@requires_db
async def test_similarity_search_respects_top_k(client) -> None:
    engine, db = await _get_session()
    try:
        results = await similarity_search(db, "sample ratio mismatch health check", top_k=2)
    finally:
        await db.close()
        await engine.dispose()

    assert len(results) <= 2
