"""Integration test fixtures.

These require a live Postgres with the pgvector extension available
(the same one docker-compose brings up, or CI's postgres service) —
tests using the `client` fixture are skipped automatically if it can't
be reached, so `pytest` still runs cleanly with no infra for the pure
unit tests in test_stats_engine.py / test_ml_engine.py / etc.

Schema is managed by Alembic (`alembic upgrade head`), not created/dropped
here — this fixture file only truncates app tables before the run so
repeated local runs against a persistent dev database (e.g. Supabase)
don't collide on unique constraints from a prior run's leftover rows.
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.api.deps import get_db
from app.core.config import get_settings
from app.db.models import *  # noqa: F401,F403 - register all models on Base.metadata
from app.main import app

settings = get_settings()

APP_TABLES = ["chat_messages", "chat_sessions", "ml_runs", "experiments", "users"]


def _db_is_reachable() -> bool:
    try:
        engine = create_engine(settings.database_url_sync, pool_pre_ping=True)
        with engine.connect():
            return True
    except Exception:
        return False


DB_AVAILABLE = _db_is_reachable()
requires_db = pytest.mark.skipif(not DB_AVAILABLE, reason="No reachable Postgres for integration tests")


@pytest.fixture(scope="session")
def _prepare_schema():
    if not DB_AVAILABLE:
        yield
        return

    engine = create_engine(settings.database_url_sync)
    with engine.begin() as conn:
        conn.execute(text(f"TRUNCATE {', '.join(APP_TABLES)} RESTART IDENTITY CASCADE"))
    yield


@pytest_asyncio.fixture
async def client(_prepare_schema):
    # A fresh engine per test (NullPool, no connection reuse) rather than the app's module-level
    # engine: pytest-asyncio gives each test function its own event loop, but asyncpg connections
    # are bound to the loop that created them — reusing a pooled connection across tests' loops
    # raises "attached to a different loop".
    test_engine = create_async_engine(settings.database_url, poolclass=NullPool)
    TestSessionLocal = async_sessionmaker(test_engine, expire_on_commit=False)

    async def _override_get_db():
        async with TestSessionLocal() as session:
            yield session

    app.dependency_overrides[get_db] = _override_get_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()
    await test_engine.dispose()
