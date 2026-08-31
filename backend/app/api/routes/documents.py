"""Lets a user upload their own documents (CSV/TXT/MD/PDF) so ABBot can answer
questions grounded in their own data, not just the curated knowledge base.
Uploaded content is chunked and embedded the same way the curated KB is, then
folded into the same retrieval pool — scoped to that user's account, so one
person's upload never surfaces in another account's answers."""

import uuid

from fastapi import APIRouter, HTTPException, UploadFile, status
from sqlalchemy import func, select

from app.api.deps import CurrentUser, DbSession
from app.db.models.user_document import UserDocument, UserDocumentChunk
from app.rag.chunking import chunk_text
from app.rag.embeddings import embed_texts
from app.schemas.document import UserDocumentContentResponse, UserDocumentResponse
from app.services.document_processing import extract_text, file_extension

router = APIRouter(prefix="/api/documents", tags=["documents"])

MAX_DOCUMENTS_PER_USER = 20


@router.get("", response_model=list[UserDocumentResponse])
async def list_documents(current_user: CurrentUser, db: DbSession) -> list[UserDocumentResponse]:
    stmt = (
        select(UserDocument).where(UserDocument.user_id == current_user.id).order_by(UserDocument.created_at.desc())
    )
    docs = (await db.execute(stmt)).scalars().all()
    return [UserDocumentResponse.model_validate(d) for d in docs]


@router.post("/upload", response_model=UserDocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile, current_user: CurrentUser, db: DbSession) -> UserDocumentResponse:
    existing_count = (
        await db.execute(select(func.count()).select_from(UserDocument).where(UserDocument.user_id == current_user.id))
    ).scalar_one()
    if existing_count >= MAX_DOCUMENTS_PER_USER:
        raise HTTPException(
            status_code=400,
            detail=f"You've reached the limit of {MAX_DOCUMENTS_PER_USER} uploaded documents. Delete one first.",
        )

    raw_bytes = await file.read()
    try:
        content = extract_text(file.filename or "upload", raw_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    document = UserDocument(
        user_id=current_user.id,
        filename=file.filename or "upload",
        file_type=file_extension(file.filename or ""),
        content=content,
    )
    db.add(document)
    await db.flush()

    chunks = chunk_text(content)
    embeddings = embed_texts(chunks)
    for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=False)):
        db.add(UserDocumentChunk(document_id=document.id, chunk_index=idx, content=chunk, embedding=embedding))

    await db.commit()
    await db.refresh(document)
    return UserDocumentResponse.model_validate(document)


@router.get("/{document_id}", response_model=UserDocumentContentResponse)
async def get_document(document_id: uuid.UUID, current_user: CurrentUser, db: DbSession) -> UserDocumentContentResponse:
    doc = await db.get(UserDocument, document_id)
    if doc is None or doc.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Document not found.")
    return UserDocumentContentResponse.model_validate(doc)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(document_id: uuid.UUID, current_user: CurrentUser, db: DbSession) -> None:
    doc = await db.get(UserDocument, document_id)
    if doc is None or doc.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Document not found.")
    await db.delete(doc)
    await db.commit()
