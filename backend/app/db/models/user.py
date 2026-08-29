from typing import TYPE_CHECKING

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDPKMixin

if TYPE_CHECKING:
    from app.db.models.chat import ChatSession
    from app.db.models.experiment import Experiment
    from app.db.models.ml_run import MLRun


class User(Base, UUIDPKMixin, TimestampMixin):
    __tablename__ = "users"

    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Which track the user signed up for. Business users are running real experiments and
    # want result interpretation / next steps; learners want to be taught the concepts.
    # Drives both UI copy and the RAG chat's system prompt — see app/rag/llm_client.py.
    persona: Mapped[str] = mapped_column(String(20), default="business", nullable=False)

    experiments: Mapped[list["Experiment"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    ml_runs: Mapped[list["MLRun"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    chat_sessions: Mapped[list["ChatSession"]] = relationship(back_populates="user", cascade="all, delete-orphan")
