from typing import TYPE_CHECKING

from sqlalchemy import String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDPKMixin

if TYPE_CHECKING:
    from app.db.models.chat import ChatSession
    from app.db.models.experiment import Experiment
    from app.db.models.metric import Metric
    from app.db.models.ml_run import MLRun
    from app.db.models.user_document import UserDocument


class User(Base, UUIDPKMixin, TimestampMixin):
    __tablename__ = "users"
    # Deliberately (email, persona), not email alone: business and learner are separate
    # products with separate data and (eventually) separate billing, so the same person
    # can hold a business account and a learner account under the same email — two
    # distinct accounts, not one account with a mutable persona flag.
    __table_args__ = (UniqueConstraint("email", "persona", name="uq_users_email_persona"),)

    email: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Which product this specific account is for. Business accounts are running real
    # experiments and want result interpretation / next steps; learner accounts are here
    # to be taught the concepts. Drives UI copy and the RAG chat's system prompt (see
    # app/rag/llm_client.py) — and, unlike a simple preference, is fixed for the account's
    # lifetime: switching tracks means signing up for the other account, not flipping a flag.
    persona: Mapped[str] = mapped_column(String(20), default="business", nullable=False)

    experiments: Mapped[list["Experiment"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    ml_runs: Mapped[list["MLRun"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    chat_sessions: Mapped[list["ChatSession"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    metrics: Mapped[list["Metric"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    documents: Mapped[list["UserDocument"]] = relationship(back_populates="user", cascade="all, delete-orphan")
