from typing import Any

from pydantic import BaseModel


class SampleDatasetSummary(BaseModel):
    key: str
    name: str
    description: str
    group_col: str
    metric_col: str
    row_count: int


class SampleDatasetDetail(SampleDatasetSummary):
    rows: list[dict[str, Any]]


class GenerateDatasetRequest(BaseModel):
    domain: str
    n_samples: int = 5000
