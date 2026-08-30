"""A reusable metrics catalog — define what "Checkout Conversion" or "Page Load
Time" means once, then reuse it by name across every future experiment instead of
re-picking raw column names each time. This is the feature Statsig's product is
actually organized around, not a cosmetic addition: real experimentation platforms
give a metric a durable identity independent of any one analysis."""

import uuid

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select

from app.api.deps import CurrentUser, DbSession
from app.db.models.metric import Metric
from app.schemas.metric import CreateMetricRequest, MetricResponse

router = APIRouter(prefix="/api/metrics", tags=["metrics"])


@router.get("", response_model=list[MetricResponse])
async def list_metrics(current_user: CurrentUser, db: DbSession) -> list[MetricResponse]:
    stmt = select(Metric).where(Metric.user_id == current_user.id).order_by(Metric.created_at.desc())
    metrics = (await db.execute(stmt)).scalars().all()
    return [MetricResponse.model_validate(m) for m in metrics]


@router.post("", response_model=MetricResponse, status_code=status.HTTP_201_CREATED)
async def create_metric(payload: CreateMetricRequest, current_user: CurrentUser, db: DbSession) -> MetricResponse:
    existing = (
        await db.execute(select(Metric).where(Metric.user_id == current_user.id, Metric.name == payload.name))
    ).scalar_one_or_none()
    if existing is not None:
        raise HTTPException(status_code=409, detail=f"A metric named '{payload.name}' already exists.")

    metric = Metric(
        user_id=current_user.id,
        name=payload.name,
        description=payload.description,
        column_name=payload.column_name,
        is_guardrail=payload.is_guardrail,
    )
    db.add(metric)
    await db.commit()
    await db.refresh(metric)
    return MetricResponse.model_validate(metric)


@router.delete("/{metric_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_metric(metric_id: uuid.UUID, current_user: CurrentUser, db: DbSession) -> None:
    metric = await db.get(Metric, metric_id)
    if metric is None or metric.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Metric not found.")
    await db.delete(metric)
    await db.commit()
