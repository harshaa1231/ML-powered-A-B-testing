from pydantic import BaseModel

from app.schemas.chat import ChatSource


class TrendPoint(BaseModel):
    week: str
    count: int
    significant: int


class AnalyticsOverviewResponse(BaseModel):
    total_experiments: int
    significant_count: int
    significance_rate: float
    experiments_this_week: int
    test_type_breakdown: dict[str, int]
    guardrail_failure_rate: float | None
    trend: list[TrendPoint]
    ai_summary: str
    sources: list[ChatSource]
