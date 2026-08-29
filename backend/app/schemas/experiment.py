import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel


class SimpleTestRequest(BaseModel):
    name: str = "Untitled experiment"
    metric_type: Literal["conversion", "continuous"]
    domain: str = "general"

    # conversion metric
    control_conversions: int | None = None
    control_total: int | None = None
    treatment_conversions: int | None = None
    treatment_total: int | None = None

    # continuous metric
    control_values: list[float] | None = None
    treatment_values: list[float] | None = None


class AdvancedTestRequest(BaseModel):
    name: str = "Untitled experiment"
    domain: str = "general"
    group_col: str
    metric_col: str
    test_type: Literal["auto", "ttest", "chi_square", "mann_whitney"] = "auto"
    rows: list[dict[str, Any]]


class ExperimentResultResponse(BaseModel):
    test_name: str
    p_value: float
    effect_size: float
    uplift_percentage: float
    is_significant: bool
    mean_control: float | None = None
    mean_treatment: float | None = None
    p_control: float | None = None
    p_treatment: float | None = None
    n_control: int | None = None
    n_treatment: int | None = None


class ExperimentResponse(BaseModel):
    id: uuid.UUID
    name: str
    mode: str
    domain: str
    test_type: str
    group_col: str | None
    metric_col: str | None
    results: dict[str, Any]
    created_at: datetime

    model_config = {"from_attributes": True}
