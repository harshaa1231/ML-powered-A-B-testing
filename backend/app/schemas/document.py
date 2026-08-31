import uuid
from datetime import datetime

from pydantic import BaseModel


class UserDocumentResponse(BaseModel):
    id: uuid.UUID
    filename: str
    file_type: str
    created_at: datetime

    model_config = {"from_attributes": True}


class UserDocumentContentResponse(UserDocumentResponse):
    content: str
