import uuid
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDPKMixin

if TYPE_CHECKING:
    from app.db.models.user import User


class Metric(Base, UUIDPKMixin, TimestampMixin):
    """A named, reusable metric definition — the thing Statsig's whole product is
    actually built around: define what "Checkout Conversion" means once, then pick
    it by name on every future experiment instead of re-describing raw column names
    each time. `column_name` is the column this metric maps to in a user's typical
    dataset export (their pipeline presumably names it consistently)."""

    __tablename__ = "metrics"
    __table_args__ = (UniqueConstraint("user_id", "name", name="uq_metrics_user_name"),)

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    column_name: Mapped[str] = mapped_column(String(255), nullable=False)
    is_guardrail: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    user: Mapped["User"] = relationship(back_populates="metrics")
