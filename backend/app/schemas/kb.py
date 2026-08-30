from pydantic import BaseModel


class KBDocumentResponse(BaseModel):
    slug: str
    title: str
    content: str
