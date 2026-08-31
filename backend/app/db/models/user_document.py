import uuid
from typing import TYPE_CHECKING

from pgvector.sqlalchemy import Vector
from sqlalchemy import ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.config import get_settings
from app.db.base import Base, TimestampMixin, UUIDPKMixin

if TYPE_CHECKING:
    from app.db.models.user import User

settings = get_settings()


class UserDocument(Base, UUIDPKMixin, TimestampMixin):
    """A document a user uploaded (CSV/TXT/MD/PDF) so ABBot can answer questions
    grounded in their own data, not just the curated knowledge base. Chunked and
    embedded the same way the curated KB is — the retrieval pipeline treats both
    as one searchable pool, scoped per user for uploads."""

    __tablename__ = "user_documents"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_type: Mapped[str] = mapped_column(String(10), nullable=False)  # "csv" | "txt" | "md" | "pdf"
    content: Mapped[str] = mapped_column(Text, nullable=False)  # extracted/summarized text

    user: Mapped["User"] = relationship(back_populates="documents")
    chunks: Mapped[list["UserDocumentChunk"]] = relationship(back_populates="document", cascade="all, delete-orphan")


class UserDocumentChunk(Base, UUIDPKMixin):
    __tablename__ = "user_document_chunks"

    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("user_documents.id", ondelete="CASCADE"), nullable=False
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(settings.embedding_dim), nullable=False)

    document: Mapped["UserDocument"] = relationship(back_populates="chunks")
