"""Lets a 'Grounded in the knowledge base' citation actually be clicked and read —
previously those source pills were plain, non-interactive text with nothing behind
them, even though the full document content already existed in the database."""

from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from app.api.deps import CurrentUser, DbSession
from app.db.models.kb_document import KBDocument
from app.schemas.kb import KBDocumentResponse

router = APIRouter(prefix="/api/kb", tags=["knowledge-base"])


@router.get("/{slug}", response_model=KBDocumentResponse)
async def get_kb_document(slug: str, current_user: CurrentUser, db: DbSession) -> KBDocumentResponse:
    doc = (await db.execute(select(KBDocument).where(KBDocument.slug == slug))).scalar_one_or_none()
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Unknown knowledge base document '{slug}'.")
    return KBDocumentResponse(slug=doc.slug, title=doc.title, content=doc.content)
