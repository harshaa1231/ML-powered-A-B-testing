"""Retrieval-augmented answer generation: KB retrieval + live experiment
context injection + Groq generation.

Mirrors the old `build_context_message` pattern from the Streamlit
prototype's hf_chat.py, but now grounds answers in retrieved KB chunks
too, not just the live experiment numbers.
"""

from __future__ import annotations

import uuid
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
    metric = results.get("metric")
    p_control, p_treatment = results.get("p_control"), results.get("p_treatment")
    mean_control, mean_treatment = results.get("mean_control"), results.get("mean_treatment")

    parts = ["The user's most recent experiment produced these results:"]
    if test_name:
        parts.append(f"- Test type: {test_name}")
    if metric:
        parts.append(f"- Metric analyzed: {metric}")
    if p is not None:
        parts.append(f"- P-value: {p:.4f}")
    if sig is not None:
        parts.append(f"- Statistically significant: {'Yes' if sig else 'No'}")
    if uplift is not None:
        parts.append(f"- Uplift (treatment vs control): {uplift:.2f}%")
    if p_control is not None and p_treatment is not None:
        parts.append(f"- Control group rate: {p_control * 100:.2f}%")
        parts.append(f"- Treatment group rate: {p_treatment * 100:.2f}%")
    if mean_control is not None and mean_treatment is not None:
        parts.append(f"- Control group mean: {mean_control:.4f}")
        parts.append(f"- Treatment group mean: {mean_treatment:.4f}")
    if n_c and n_t:
        parts.append(f"- Sample sizes: {n_c:,} control, {n_t:,} treatment")
    guardrails = results.get("guardrails")
    if guardrails:
        parts.append("- Guardrail metrics also tested:")
        for g in guardrails:
            g_metric = g.get("metric", "unknown")
            g_sig = "significant" if g.get("is_significant") else "not significant"
            parts.append(f"  - {g_metric}: p={g.get('p_value', float('nan')):.4f} ({g_sig})")
    parts.append(
        "These are the ONLY numbers that exist for this experiment. Only cite figures listed above — "
        "never invent additional metrics, time windows, or numeric breakdowns that aren't given here, "
        "even if they sound plausible or you recall similar-sounding published statistics."
    )
    return "\n".join(parts)


def build_kb_context(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return ""
    blocks = [f"[{c.title}]\n{c.content}" for c in chunks]
    return (
        "Relevant reference material (this may include documents the user uploaded themselves, "
        "not just the curated knowledge base — treat both as equally valid sources):\n\n"
        + "\n\n---\n\n".join(blocks)
    )


async def answer_question(
    db: AsyncSession,
    question: str,
    history: list[dict[str, str]],
    experiment_results: dict[str, Any] | None = None,
    persona: Persona | None = None,
    user_id: uuid.UUID | None = None,
) -> tuple[str, list[RetrievedChunk]]:
    retrieved = await similarity_search(db, question, top_k=settings.rag_top_k, user_id=user_id)

    context_parts = [p for p in (build_kb_context(retrieved), build_experiment_context(experiment_results)) if p]
    context_message = "\n\n".join(context_parts)

    messages = list(history)
    if context_message:
        messages.append({"role": "system", "content": context_message})
    messages.append({"role": "user", "content": question})

    answer = await chat_completion(messages, persona=persona)
    return answer, retrieved
