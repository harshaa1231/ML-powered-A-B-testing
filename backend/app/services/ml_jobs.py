"""Background training jobs for the ML Model Studio.

Training runs via FastAPI BackgroundTasks (in a worker thread, since
scikit-learn training is CPU-bound and blocking) with progress tracked
as a status row in Postgres that the frontend polls. This avoids
pulling in Redis/Celery for a workload this small while still
demonstrating a real async job pattern.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import pandas as pd

from app.db.models.ml_run import MLRun, MLRunStatus
from app.db.session import AsyncSessionLocal
from app.services.ml_engine import UniversalMLEngine
from app.storage import get_storage


def _train_sync(rows: list[dict[str, Any]], target_col: str, group_col: str | None, model_type: str, task: str) -> tuple[dict[str, Any], bytes]:
    df = pd.DataFrame(rows)
    engine = UniversalMLEngine()

    if task == "uplift":
        if not group_col:
            raise ValueError("group_col is required for uplift modeling.")
        results = engine.train_uplift_model(df, target_col=target_col, treatment_col=group_col)
    else:
        results = engine.train_model(df, target_col=target_col, group_col=group_col, model_type=model_type)

    return results, engine.to_bytes()


async def run_training_job(
    ml_run_id: uuid.UUID,
    rows: list[dict[str, Any]],
    target_col: str,
    group_col: str | None,
    model_type: str,
    task: str,
) -> None:
    storage = get_storage()

    async with AsyncSessionLocal() as db:
        ml_run = await db.get(MLRun, ml_run_id)
        if ml_run is None:
            return
        ml_run.status = MLRunStatus.RUNNING
        await db.commit()

    try:
        results, model_bytes = await asyncio.to_thread(_train_sync, rows, target_col, group_col, model_type, task)
        model_key = storage.new_key(prefix=f"models/{ml_run_id}")
        storage.save_bytes(model_key, model_bytes)

        async with AsyncSessionLocal() as db:
            ml_run = await db.get(MLRun, ml_run_id)
            if ml_run is not None:
                ml_run.status = MLRunStatus.DONE
                ml_run.results = results
                ml_run.model_path = model_key
                await db.commit()
    except Exception as exc:  # noqa: BLE001 - persist any training failure for the frontend to display
        async with AsyncSessionLocal() as db:
            ml_run = await db.get(MLRun, ml_run_id)
            if ml_run is not None:
                ml_run.status = MLRunStatus.FAILED
                ml_run.error_message = str(exc)
                await db.commit()
