import uuid

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from sqlalchemy import select

from app.api.deps import CurrentUser, DbSession
from app.db.models.ml_run import MLRun, MLRunStatus
from app.schemas.ml import MLRunResponse, PredictRequest, PredictResponse, TrainModelRequest
from app.services.ml_engine import UniversalMLEngine
from app.storage import get_storage

router = APIRouter(prefix="/api/ml", tags=["ml"])


@router.post("/train", response_model=MLRunResponse, status_code=status.HTTP_202_ACCEPTED)
async def train_model(
    payload: TrainModelRequest, current_user: CurrentUser, db: DbSession, background_tasks: BackgroundTasks
) -> MLRunResponse:
    if not payload.rows:
        raise HTTPException(status_code=400, detail="No rows provided.")

    ml_run = MLRun(
        user_id=current_user.id,
        experiment_id=payload.experiment_id,
        task_type=payload.task,
        status=MLRunStatus.PENDING,
        target_col=payload.target_col,
        group_col=payload.group_col,
        model_type=payload.model_type,
    )
    db.add(ml_run)
    await db.commit()
    await db.refresh(ml_run)

    from app.services.ml_jobs import run_training_job  # local import avoids a circular import at module load

    background_tasks.add_task(
        run_training_job, ml_run.id, payload.rows, payload.target_col, payload.group_col, payload.model_type, payload.task
    )

    return MLRunResponse.model_validate(ml_run)


@router.get("/runs", response_model=list[MLRunResponse])
async def list_runs(current_user: CurrentUser, db: DbSession) -> list[MLRunResponse]:
    stmt = select(MLRun).where(MLRun.user_id == current_user.id).order_by(MLRun.created_at.desc())
    runs = (await db.execute(stmt)).scalars().all()
    return [MLRunResponse.model_validate(r) for r in runs]


@router.get("/runs/{run_id}", response_model=MLRunResponse)
async def get_run(run_id: uuid.UUID, current_user: CurrentUser, db: DbSession) -> MLRunResponse:
    ml_run = await db.get(MLRun, run_id)
    if ml_run is None or ml_run.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="ML run not found.")
    return MLRunResponse.model_validate(ml_run)


@router.post("/predict", response_model=PredictResponse)
async def predict(payload: PredictRequest, current_user: CurrentUser, db: DbSession) -> PredictResponse:
    ml_run = await db.get(MLRun, payload.ml_run_id)
    if ml_run is None or ml_run.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="ML run not found.")
    if ml_run.status != MLRunStatus.DONE or not ml_run.model_path:
        raise HTTPException(status_code=400, detail=f"Model is not ready (status: {ml_run.status.value}).")
    if not payload.rows:
        raise HTTPException(status_code=400, detail="No rows provided.")

    storage = get_storage()
    model_bytes = storage.read_bytes(ml_run.model_path)
    engine = UniversalMLEngine.from_bytes(model_bytes)

    df = pd.DataFrame(payload.rows)
    try:
        if ml_run.task_type == "uplift":
            predictions = engine.predict_uplift(df)
        else:
            predictions = engine.predict(df)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return PredictResponse(predictions=[float(p) for p in predictions])
