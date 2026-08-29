from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException

from app.api.deps import CurrentUser
from app.schemas.dataset import GenerateDatasetRequest, SampleDatasetDetail, SampleDatasetSummary
from app.services.data_generator import DOMAINS, EnhancedDataGenerator
from app.services.ml_engine import UniversalMLEngine
from app.services.sample_data import SampleDataGenerator

router = APIRouter(prefix="/api/datasets", tags=["datasets"])

MAX_PREVIEW_ROWS = 2000


@router.get("/samples", response_model=list[SampleDatasetSummary])
async def list_sample_datasets(current_user: CurrentUser) -> list[SampleDatasetSummary]:
    samples = SampleDataGenerator.get_all_samples()
    return [
        SampleDatasetSummary(
            key=key,
            name=sample["name"],
            description=sample["description"],
            group_col=sample["group_col"],
            metric_col=sample["metric_col"],
            row_count=len(sample["df"]),
        )
        for key, sample in samples.items()
    ]


@router.get("/samples/{key}", response_model=SampleDatasetDetail)
async def get_sample_dataset(key: str, current_user: CurrentUser) -> SampleDatasetDetail:
    samples = SampleDataGenerator.get_all_samples()
    if key not in samples:
        raise HTTPException(status_code=404, detail=f"Unknown sample dataset '{key}'.")

    sample = samples[key]
    df = sample["df"]
    return SampleDatasetDetail(
        key=key,
        name=sample["name"],
        description=sample["description"],
        group_col=sample["group_col"],
        metric_col=sample["metric_col"],
        row_count=len(df),
        rows=df.head(MAX_PREVIEW_ROWS).to_dict(orient="records"),
    )


@router.get("/generator/domains", response_model=list[str])
async def list_generator_domains(current_user: CurrentUser) -> list[str]:
    return DOMAINS


@router.post("/generator/generate")
async def generate_synthetic_dataset(payload: GenerateDatasetRequest, current_user: CurrentUser) -> dict[str, Any]:
    if payload.domain not in DOMAINS:
        raise HTTPException(status_code=400, detail=f"Unknown domain '{payload.domain}'. Choose from {DOMAINS}.")
    if not (100 <= payload.n_samples <= 200_000):
        raise HTTPException(status_code=400, detail="n_samples must be between 100 and 200,000.")

    df = EnhancedDataGenerator.generate_domain(payload.domain, payload.n_samples)
    return {
        "domain": payload.domain,
        "row_count": len(df),
        "rows": df.head(MAX_PREVIEW_ROWS).to_dict(orient="records"),
        "truncated": len(df) > MAX_PREVIEW_ROWS,
    }


@router.post("/detect-columns")
async def detect_columns(rows: list[dict[str, Any]], current_user: CurrentUser) -> dict[str, Any]:
    if not rows:
        raise HTTPException(status_code=400, detail="No rows provided.")
    df = pd.DataFrame(rows)
    engine = UniversalMLEngine()
    return engine.auto_detect_columns(df)
