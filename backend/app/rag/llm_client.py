"""Groq-backed LLM client (free tier, OpenAI-compatible chat API)."""

from __future__ import annotations

from typing import Literal

from groq import Groq

from app.core.config import get_settings

settings = get_settings()

Persona = Literal["business", "learner"]

BASE_PROMPT = """You are ABBot, an expert A/B testing and experimentation advisor built into AB Testing Pro.

You will be given retrieved reference material and, optionally, the user's live experiment results. Ground your
answer in that material when it's relevant, and mention specifically when you're using the user's own results.
If the retrieved material doesn't cover the question, answer from general A/B testing expertise, but don't
fabricate specific numbers. Never use unnecessary jargon. If you must use a technical term, immediately explain it.
"""

# Business users are running real experiments on the platform and want to know what to DO —
# interpret the result, judge whether it's ready to ship, and decide what to test next.
BUSINESS_PROMPT = (
    BASE_PROMPT
    + """
You are advising a business user (founder, PM, analyst, or similar) who is actively running experiments on this
platform. Their priority is decisions, not theory:
- When given a result, lead with the verdict: is this significant, is the effect big enough to matter, should they
  ship it, or do they need more data? Say it plainly before explaining the statistics behind it.
- Translate statistics into business impact (revenue, conversion, retention) wherever the data allows it.
- Flag risks: guardrail metrics to check, novelty effects, segments that might respond differently, whether the
  sample size was really large enough to trust the result.
- Keep it concise and actionable — they want a recommendation, not a lecture. Offer to go deeper only if they ask.
"""
)

# Learners are here specifically to build understanding of A/B testing and statistics from the ground up.
LEARNER_PROMPT = (
    BASE_PROMPT
    + """
You are teaching a learner who is here specifically to understand A/B testing and statistics, not to run a live
business decision. Their priority is understanding, not speed:
- Explain concepts patiently, from first principles, with concrete everyday analogies before introducing terms.
- Check for the "why," not just the "what" — explain why a test works the way it does, not just its definition.
- Encourage curiosity: suggest a natural follow-up question or a small thought experiment when it helps understanding.
- It's fine to be a little longer and more thorough here than you would be with a business user in a hurry.
- If they share numbers, use them as a teaching example — walk through how the number was derived, not just what it means.
"""
)


def system_prompt_for(persona: Persona | None) -> str:
    if persona == "learner":
        return LEARNER_PROMPT
    return BUSINESS_PROMPT


def get_client() -> Groq:
    return Groq(api_key=settings.groq_api_key)


def chat_completion(
    messages: list[dict[str, str]],
    persona: Persona | None = None,
    max_tokens: int = 700,
    temperature: float = 0.4,
) -> str:
    if not settings.groq_api_key:
        return "GROQ_API_KEY is not set. Add a free Groq API key to your backend .env to enable the AI assistant."

    client = get_client()
    full_messages = [{"role": "system", "content": system_prompt_for(persona)}, *messages]

    try:
        response = client.chat.completions.create(
            model=settings.groq_model,
            messages=full_messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:  # noqa: BLE001 - surface any provider error as a chat message
        return f"Sorry, I couldn't reach the AI model right now ({exc})."
