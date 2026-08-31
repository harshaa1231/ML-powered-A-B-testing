import uuid
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import CurrentUser, DbSession
from app.db.models.experiment import Experiment
from app.db.models.user import User
from app.rag.retriever import answer_question
from app.schemas.experiment import AdvancedTestRequest, ExperimentResponse, SimpleTestRequest, UpdateDecisionRequest
from app.services.experiment_analysis import run_ab_analysis
from app.services.stats_engine import StatisticalTester

router = APIRouter(prefix="/api/experiments", tags=["experiments"])


async def _generate_ai_summary(db: AsyncSession, results: dict[str, Any], user: User) -> str:
    """RAG-grounded summary: goes through the same retrieval pipeline chat uses,
    not a bare LLM call, so it cites the KB doc relevant to this specific result
    (test type, health-check outcome) rather than freelancing from general knowledge.
    """
    test_name = results.get("test_name", "this test")
    parts = [
        f"In 2-4 sentences, summarize this {test_name} result: is it statistically significant, "
        "does the effect size actually matter, and what should I do next?"
    ]

    srm = results.get("health_checks", {}).get("sample_ratio_mismatch")
    if srm and not srm.get("passed", True):
        parts.append(
            "Also flag that the sample ratio mismatch health check failed, which could mean broken randomization."
        )

    significant_guardrails = [g["metric"] for g in results.get("guardrails", []) if g.get("is_significant")]
    if significant_guardrails:
        parts.append(
            "Also note that these guardrail metrics showed a statistically significant change and should be "
            f"reviewed: {', '.join(significant_guardrails)}."
        )

    query = " ".join(parts)
    answer, _ = await answer_question(db, query, history=[], experiment_results=results, persona=user.persona, user_id=user.id)
    return answer


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

    results["health_checks"] = {
        "sample_ratio_mismatch": StatisticalTester.sample_ratio_mismatch(
            results.get("n_control", 0), results.get("n_treatment", 0)
        )
    }
    results["ai_summary"] = await _generate_ai_summary(db, results, current_user)

    experiment = Experiment(
        user_id=current_user.id,
        name=payload.name,
        mode="simple",
        domain=payload.domain,
        test_type=test_type,
        hypothesis=payload.hypothesis,
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
        results = run_ab_analysis(
            df, payload.group_col, payload.metric_col, payload.test_type, payload.domain, payload.guardrail_cols
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    results["ai_summary"] = await _generate_ai_summary(db, results, current_user)

    experiment = Experiment(
        user_id=current_user.id,
        name=payload.name,
        mode="advanced",
        domain=payload.domain,
        test_type=results.get("test_name", payload.test_type),
        hypothesis=payload.hypothesis,
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


@router.patch("/{experiment_id}/decision", response_model=ExperimentResponse)
async def update_decision(
    experiment_id: uuid.UUID, payload: UpdateDecisionRequest, current_user: CurrentUser, db: DbSession
) -> ExperimentResponse:
    """Records what actually happened after the result came in — an experiment is a
    system of record with a real outcome, not just a number read once and forgotten."""
    stmt = select(Experiment).where(Experiment.id == experiment_id, Experiment.user_id == current_user.id)
    experiment = (await db.execute(stmt)).scalar_one_or_none()
    if experiment is None:
        raise HTTPException(status_code=404, detail="Experiment not found.")
    experiment.decision = payload.decision
    await db.commit()
    await db.refresh(experiment)
    return ExperimentResponse.model_validate(experiment)
