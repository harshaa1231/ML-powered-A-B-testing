import uuid

import pandas as pd
from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select

from app.api.deps import CurrentUser, DbSession
from app.db.models.experiment import Experiment
from app.schemas.experiment import AdvancedTestRequest, ExperimentResponse, SimpleTestRequest
from app.services.experiment_analysis import run_ab_analysis
from app.services.stats_engine import StatisticalTester

router = APIRouter(prefix="/api/experiments", tags=["experiments"])


@router.post("/simple", response_model=ExperimentResponse, status_code=status.HTTP_201_CREATED)
async def run_simple_test(payload: SimpleTestRequest, current_user: CurrentUser, db: DbSession) -> ExperimentResponse:
    tester = StatisticalTester()

    if payload.metric_type == "conversion":
        if None in (payload.control_conversions, payload.control_total, payload.treatment_conversions, payload.treatment_total):
            raise HTTPException(status_code=400, detail="Conversion counts and totals are required for a conversion metric.")
        results = tester.two_proportion_test(
            payload.control_conversions, payload.control_total, payload.treatment_conversions, payload.treatment_total
        )
        test_type = "two_proportion_z"
    else:
        if not payload.control_values or not payload.treatment_values:
            raise HTTPException(status_code=400, detail="control_values and treatment_values are required for a continuous metric.")
        results = tester.independent_ttest(payload.control_values, payload.treatment_values)
        test_type = "welch_ttest"

    experiment = Experiment(
        user_id=current_user.id,
        name=payload.name,
        mode="simple",
        domain=payload.domain,
        test_type=test_type,
        group_col=None,
        metric_col=None,
        results=results,
    )
    db.add(experiment)
    await db.commit()
    await db.refresh(experiment)
    return ExperimentResponse.model_validate(experiment)


@router.post("/advanced", response_model=ExperimentResponse, status_code=status.HTTP_201_CREATED)
async def run_advanced_test(payload: AdvancedTestRequest, current_user: CurrentUser, db: DbSession) -> ExperimentResponse:
    if not payload.rows:
        raise HTTPException(status_code=400, detail="No rows provided.")

    df = pd.DataFrame(payload.rows)
    if payload.group_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"Group column '{payload.group_col}' not found in data.")
    if payload.metric_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"Metric column '{payload.metric_col}' not found in data.")

    try:
        results = run_ab_analysis(df, payload.group_col, payload.metric_col, payload.test_type, payload.domain)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    experiment = Experiment(
        user_id=current_user.id,
        name=payload.name,
        mode="advanced",
        domain=payload.domain,
        test_type=results.get("test_name", payload.test_type),
        group_col=payload.group_col,
        metric_col=payload.metric_col,
        results=results,
    )
    db.add(experiment)
    await db.commit()
    await db.refresh(experiment)
    return ExperimentResponse.model_validate(experiment)


@router.get("", response_model=list[ExperimentResponse])
async def list_experiments(current_user: CurrentUser, db: DbSession) -> list[ExperimentResponse]:
    stmt = select(Experiment).where(Experiment.user_id == current_user.id).order_by(Experiment.created_at.desc())
    experiments = (await db.execute(stmt)).scalars().all()
    return [ExperimentResponse.model_validate(e) for e in experiments]


@router.get("/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment(experiment_id: uuid.UUID, current_user: CurrentUser, db: DbSession) -> ExperimentResponse:
    stmt = select(Experiment).where(Experiment.id == experiment_id, Experiment.user_id == current_user.id)
    experiment = (await db.execute(stmt)).scalar_one_or_none()
    if experiment is None:
        raise HTTPException(status_code=404, detail="Experiment not found.")
    return ExperimentResponse.model_validate(experiment)
