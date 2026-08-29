import uuid
from datetime import datetime

from pydantic import BaseModel


class ChatMessageRequest(BaseModel):
    session_id: uuid.UUID | None = None
    message: str
    experiment_id: uuid.UUID | None = None  # ground the answer in this experiment's live results


class ChatSource(BaseModel):
    slug: str
    title: str
    similarity: float


class ChatMessageResponse(BaseModel):
    session_id: uuid.UUID
    role: str
    content: str
    sources: list[ChatSource]


class ChatHistoryMessage(BaseModel):
    role: str
    content: str
    created_at: datetime

    model_config = {"from_attributes": True}
