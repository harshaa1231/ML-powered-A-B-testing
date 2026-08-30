from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from sqlalchemy import select

from app.api.routes import analytics, auth, chat, datasets, experiments, kb, metrics, ml, practice
from app.core.config import get_settings
from app.core.limiter import limiter
from app.db.models.kb_document import KBDocument
from app.db.session import AsyncSessionLocal
from app.rag.ingest import ingest_knowledge_base

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with AsyncSessionLocal() as db:
        existing = (await db.execute(select(KBDocument).limit(1))).scalar_one_or_none()
        if existing is None:
            await ingest_knowledge_base(db)
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(experiments.router)
app.include_router(datasets.router)
app.include_router(ml.router)
app.include_router(chat.router)
app.include_router(practice.router)
app.include_router(analytics.router)
app.include_router(kb.router)
app.include_router(metrics.router)


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
