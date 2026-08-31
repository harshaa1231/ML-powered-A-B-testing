"""Practice Lab: RAG-grounded feedback on a learner's stated conclusion.

Uses the same `answer_question()` retrieval pipeline as chat and the
experiment AI summary — a learner's practice feedback cites the KB doc
relevant to their specific scenario and reasoning, not a static canned
answer.
"""

from fastapi import APIRouter

from app.api.deps import CurrentUser, DbSession
from app.rag.retriever import answer_question
from app.schemas.chat import ChatSource
from app.schemas.practice import PracticeFeedbackRequest, PracticeFeedbackResponse

router = APIRouter(prefix="/api/practice", tags=["practice"])


@router.post("/feedback", response_model=PracticeFeedbackResponse)
async def practice_feedback(
    payload: PracticeFeedbackRequest, current_user: CurrentUser, db: DbSession
) -> PracticeFeedbackResponse:
    query = (
        f"A learner analyzed a practice scenario about {payload.scenario_name} and concluded: "
        f'"{payload.learner_conclusion}". Given the actual computed result, give them personalized feedback: '
        "were they right, what did they miss, and what's the correct takeaway?"
    )

    answer, retrieved = await answer_question(
        db, query, history=[], experiment_results=payload.results, persona=current_user.persona, user_id=current_user.id
    )
    sources = [ChatSource(slug=c.slug, title=c.title, similarity=round(c.similarity, 3)) for c in retrieved]

    return PracticeFeedbackResponse(feedback=answer, sources=sources)
