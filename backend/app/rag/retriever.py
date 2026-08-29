"""Retrieval-augmented answer generation: KB retrieval + live experiment
context injection + Groq generation.

Mirrors the old `build_context_message` pattern from the Streamlit
prototype's hf_chat.py, but now grounds answers in retrieved KB chunks
too, not just the live experiment numbers.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.rag.llm_client import Persona, chat_completion
from app.rag.vector_store import RetrievedChunk, similarity_search

settings = get_settings()


def build_experiment_context(results: dict[str, Any] | None) -> str:
    if not results:
        return ""

    p = results.get("p_value")
    uplift = results.get("uplift_percentage")
    sig = results.get("is_significant")
    n_c = results.get("n_control")
    n_t = results.get("n_treatment")
    test_name = results.get("test_name", "")

    parts = ["The user's most recent experiment produced these results:"]
    if test_name:
        parts.append(f"- Test type: {test_name}")
    if p is not None:
        parts.append(f"- P-value: {p:.4f}")
    if sig is not None:
        parts.append(f"- Statistically significant: {'Yes' if sig else 'No'}")
    if uplift is not None:
        parts.append(f"- Uplift (treatment vs control): {uplift:.2f}%")
    if n_c and n_t:
        parts.append(f"- Sample sizes: {n_c:,} control, {n_t:,} treatment")
    parts.append("Tailor your answer to these specific numbers when the question relates to them.")
    return "\n".join(parts)


def build_kb_context(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return ""
    blocks = [f"[{c.title}]\n{c.content}" for c in chunks]
    return "Relevant reference material:\n\n" + "\n\n---\n\n".join(blocks)


async def answer_question(
    db: AsyncSession,
    question: str,
    history: list[dict[str, str]],
    experiment_results: dict[str, Any] | None = None,
    persona: Persona | None = None,
) -> tuple[str, list[RetrievedChunk]]:
    retrieved = await similarity_search(db, question, top_k=settings.rag_top_k)

    context_parts = [p for p in (build_kb_context(retrieved), build_experiment_context(experiment_results)) if p]
    context_message = "\n\n".join(context_parts)

    messages = list(history)
    if context_message:
        messages.append({"role": "system", "content": context_message})
    messages.append({"role": "user", "content": question})

    answer = chat_completion(messages, persona=persona)
    return answer, retrieved
