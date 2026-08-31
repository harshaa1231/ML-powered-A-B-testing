import uuid

from fastapi import APIRouter, HTTPException
from sqlalchemy import select

from app.api.deps import CurrentUser, DbSession
from app.db.models.chat import ChatMessage, ChatSession
from app.db.models.experiment import Experiment
from app.rag.retriever import answer_question
from app.schemas.chat import (
    ChatHistoryMessage,
    ChatMessageRequest,
    ChatMessageResponse,
    ChatSessionHistoryResponse,
    ChatSource,
)

router = APIRouter(prefix="/api/chat", tags=["chat"])

# Trimmed from 12: with chat history now persisting and being resent on every
# message (see useChatSession's history restore), each extra message here is
# real input-token cost against Groq's shared, app-wide 8000-tokens/minute free
# tier ceiling. 8 messages (4 back-and-forth turns) is still enough context for
# coherent follow-ups without eating into the budget a long, thorough answer needs.
MAX_HISTORY_MESSAGES = 8


@router.post("/message", response_model=ChatMessageResponse)
async def send_message(payload: ChatMessageRequest, current_user: CurrentUser, db: DbSession) -> ChatMessageResponse:
    if payload.session_id:
        session = await db.get(ChatSession, payload.session_id)
        if session is None or session.user_id != current_user.id:
            raise HTTPException(status_code=404, detail="Chat session not found.")
    else:
        session = ChatSession(user_id=current_user.id, title=payload.message[:60])
        db.add(session)
        await db.flush()

    history_stmt = (
        select(ChatMessage)
        .where(ChatMessage.session_id == session.id)
        .order_by(ChatMessage.created_at.desc())
        .limit(MAX_HISTORY_MESSAGES)
    )
    history_rows = list(reversed((await db.execute(history_stmt)).scalars().all()))
    history = [{"role": m.role, "content": m.content} for m in history_rows]

    experiment_results = None
    if payload.experiment_id:
        experiment = await db.get(Experiment, payload.experiment_id)
        if experiment is not None and experiment.user_id == current_user.id:
            experiment_results = experiment.results

    answer, retrieved = await answer_question(
        db, payload.message, history, experiment_results, persona=current_user.persona, user_id=current_user.id
    )

    sources = [ChatSource(slug=c.slug, title=c.title, similarity=round(c.similarity, 3)) for c in retrieved]

    db.add(ChatMessage(session_id=session.id, role="user", content=payload.message))
    db.add(
        ChatMessage(
            session_id=session.id,
            role="assistant",
            content=answer,
            sources=[s.model_dump() for s in sources],
        )
    )
    await db.commit()

    return ChatMessageResponse(session_id=session.id, role="assistant", content=answer, sources=sources)


@router.get("/sessions/latest", response_model=ChatSessionHistoryResponse)
async def get_latest_session(current_user: CurrentUser, db: DbSession) -> ChatSessionHistoryResponse:
    """Lets the chat page and the ABBot widget resume a conversation instead of
    starting blank on every page load / after logging back in — previously nothing
    ever fetched past history, even though it was persisted correctly all along."""
    session = (
        await db.execute(
            select(ChatSession).where(ChatSession.user_id == current_user.id).order_by(ChatSession.created_at.desc()).limit(1)
        )
    ).scalar_one_or_none()
    if session is None:
        return ChatSessionHistoryResponse(session_id=None, messages=[])

    stmt = select(ChatMessage).where(ChatMessage.session_id == session.id).order_by(ChatMessage.created_at)
    messages = (await db.execute(stmt)).scalars().all()
    return ChatSessionHistoryResponse(session_id=session.id, messages=[ChatHistoryMessage.model_validate(m) for m in messages])


@router.get("/sessions/{session_id}/history", response_model=list[ChatHistoryMessage])
async def get_history(session_id: uuid.UUID, current_user: CurrentUser, db: DbSession) -> list[ChatHistoryMessage]:
    session = await db.get(ChatSession, session_id)
    if session is None or session.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Chat session not found.")

    stmt = select(ChatMessage).where(ChatMessage.session_id == session_id).order_by(ChatMessage.created_at)
    messages = (await db.execute(stmt)).scalars().all()
    return [ChatHistoryMessage.model_validate(m) for m in messages]
