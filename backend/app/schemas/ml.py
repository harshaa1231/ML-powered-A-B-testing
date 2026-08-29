import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel


class TrainModelRequest(BaseModel):
    experiment_id: uuid.UUID | None = None
    rows: list[dict[str, Any]]
    target_col: str
    group_col: str | None = None
    model_type: Literal["auto", "classification", "regression"] = "auto"
    task: Literal["predictive", "uplift"] = "predictive"


class PredictRequest(BaseModel):
    ml_run_id: uuid.UUID
    rows: list[dict[str, Any]]


class MLRunResponse(BaseModel):
    id: uuid.UUID
    task_type: str
    status: str
    target_col: str
    group_col: str | None
    model_type: str
    results: dict[str, Any] | None
    error_message: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class PredictResponse(BaseModel):
    predictions: list[float]
