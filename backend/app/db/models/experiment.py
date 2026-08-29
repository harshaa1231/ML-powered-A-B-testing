import uuid
from typing import TYPE_CHECKING

from sqlalchemy import JSON, ForeignKey, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDPKMixin

if TYPE_CHECKING:
    from app.db.models.user import User


class Experiment(Base, UUIDPKMixin, TimestampMixin):
    __tablename__ = "experiments"

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    mode: Mapped[str] = mapped_column(String(20), nullable=False)  # "simple" | "advanced"
    domain: Mapped[str] = mapped_column(String(50), default="general")
    test_type: Mapped[str] = mapped_column(String(50), nullable=False)
    group_col: Mapped[str | None] = mapped_column(String(255), nullable=True)
    metric_col: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Full result payload (p_value, effect_size, uplift_percentage, is_significant, etc.)
    results: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)

    user: Mapped["User"] = relationship(back_populates="experiments")
