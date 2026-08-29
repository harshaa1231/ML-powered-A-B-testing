"""Program Analytics — Statsig's "Product Analytics," scoped honestly.

Not a general-purpose event-tracking product (that needs an ingestion
pipeline there's no reason to build here); real SQL aggregation over
the `experiments` table we already have, plus a RAG-grounded trends
narrative (a fourth `answer_question()` surface) so it reads as a
takeaway, not a wall of numbers with no story.
"""

from datetime import UTC, datetime, timedelta

from fastapi import APIRouter
from sqlalchemy import select

from app.api.deps import CurrentUser, DbSession
from app.db.models.experiment import Experiment
from app.rag.retriever import answer_question
from app.schemas.analytics import AnalyticsOverviewResponse, TrendPoint
from app.schemas.chat import ChatSource

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


@router.get("/overview", response_model=AnalyticsOverviewResponse)
async def analytics_overview(current_user: CurrentUser, db: DbSession) -> AnalyticsOverviewResponse:
    stmt = select(Experiment).where(Experiment.user_id == current_user.id).order_by(Experiment.created_at)
    experiments = list((await db.execute(stmt)).scalars().all())

    total = len(experiments)
    if total == 0:
        return AnalyticsOverviewResponse(
            total_experiments=0,
            significant_count=0,
            significance_rate=0.0,
            experiments_this_week=0,
            test_type_breakdown={},
            guardrail_failure_rate=None,
            trend=[],
            ai_summary="No experiments yet — run your first test to see program trends here.",
            sources=[],
        )

    significant_count = sum(1 for e in experiments if e.results.get("is_significant"))
    week_ago = datetime.now(UTC) - timedelta(days=7)
    experiments_this_week = sum(1 for e in experiments if e.created_at >= week_ago)

    test_type_breakdown: dict[str, int] = {}
    for e in experiments:
        test_type_breakdown[e.test_type] = test_type_breakdown.get(e.test_type, 0) + 1

    guardrail_experiments = [e for e in experiments if e.results.get("guardrails")]
    guardrail_failure_rate = (
        sum(1 for e in guardrail_experiments if any(g.get("is_significant") for g in e.results["guardrails"]))
        / len(guardrail_experiments)
        if guardrail_experiments
        else None
    )

    trend_buckets: dict[str, dict[str, int]] = {}
    for e in experiments:
        week_key = e.created_at.strftime("%Y-W%W")
        bucket = trend_buckets.setdefault(week_key, {"count": 0, "significant": 0})
        bucket["count"] += 1
        if e.results.get("is_significant"):
            bucket["significant"] += 1
    trend = [TrendPoint(week=k, **v) for k, v in sorted(trend_buckets.items())]

    query_parts = [
        f"A business user's experimentation program has run {total} experiments, {significant_count} of them "
        f"statistically significant ({significant_count / total * 100:.0f}%).",
        f"Test type breakdown: {test_type_breakdown}.",
    ]
    if guardrail_failure_rate is not None:
        query_parts.append(
            f"Guardrail metrics flagged a concern in {guardrail_failure_rate * 100:.0f}% of experiments that tracked them."
        )
    query_parts.append(
        "In 2-3 sentences, summarize what this trend says about their experimentation program and suggest one thing to focus on next."
    )

    answer, retrieved = await answer_question(db, " ".join(query_parts), history=[], persona=current_user.persona)
    sources = [ChatSource(slug=c.slug, title=c.title, similarity=round(c.similarity, 3)) for c in retrieved]

    return AnalyticsOverviewResponse(
        total_experiments=total,
        significant_count=significant_count,
        significance_rate=significant_count / total,
        experiments_this_week=experiments_this_week,
        test_type_breakdown=test_type_breakdown,
        guardrail_failure_rate=guardrail_failure_rate,
        trend=trend,
        ai_summary=answer,
        sources=sources,
    )
