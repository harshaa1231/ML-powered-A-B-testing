import enum
import uuid
from typing import TYPE_CHECKING

from sqlalchemy import JSON, Enum, ForeignKey, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDPKMixin

if TYPE_CHECKING:
    from app.db.models.user import User


class MLRunStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class MLRun(Base, UUIDPKMixin, TimestampMixin):
    __tablename__ = "ml_runs"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    # SET NULL rather than CASCADE: a trained model is still useful even if the experiment
    # that spawned it is later deleted.
    experiment_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("experiments.id", ondelete="SET NULL"), nullable=True
    )

    task_type: Mapped[str] = mapped_column(String(20), nullable=False)  # "train" | "uplift"
    status: Mapped[MLRunStatus] = mapped_column(
        Enum(MLRunStatus, name="ml_run_status"), default=MLRunStatus.PENDING, nullable=False
    )

    target_col: Mapped[str] = mapped_column(String(255), nullable=False)
    group_col: Mapped[str | None] = mapped_column(String(255), nullable=True)
    model_type: Mapped[str] = mapped_column(String(50), default="auto")

    results: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    model_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    error_message: Mapped[str | None] = mapped_column(String(2000), nullable=True)

    user: Mapped["User"] = relationship(back_populates="ml_runs")
