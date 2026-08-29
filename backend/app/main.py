from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select

from app.api.routes import analytics, auth, chat, datasets, experiments, ml, practice
from app.core.config import get_settings
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


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
