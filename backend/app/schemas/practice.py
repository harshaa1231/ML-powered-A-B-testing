from typing import Any

from pydantic import BaseModel

from app.schemas.chat import ChatSource


class PracticeFeedbackRequest(BaseModel):
    scenario_name: str
    learner_conclusion: str
    results: dict[str, Any]  # the real computed result from /api/experiments/advanced


class PracticeFeedbackResponse(BaseModel):
    feedback: str
    sources: list[ChatSource]
