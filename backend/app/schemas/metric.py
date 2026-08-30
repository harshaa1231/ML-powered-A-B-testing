import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class CreateMetricRequest(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    description: str | None = None
    column_name: str = Field(min_length=1, max_length=255)
    is_guardrail: bool = False


class MetricResponse(BaseModel):
    id: uuid.UUID
    name: str
    description: str | None
    column_name: str
    is_guardrail: bool
    created_at: datetime

    model_config = {"from_attributes": True}
