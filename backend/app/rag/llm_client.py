"""Groq-backed LLM client (free tier, OpenAI-compatible chat API)."""

from __future__ import annotations

import asyncio
from typing import Literal

from groq import Groq, RateLimitError

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


def _parse_reset_seconds(header_value: str | None, default: float = 2.0) -> float:
    """Groq returns e.g. '742ms' or '21.57s' in its rate-limit reset headers."""
    if not header_value:
        return default
    value = header_value.strip()
    try:
        if value.endswith("ms"):
            return float(value[:-2]) / 1000
        if value.endswith("s"):
            return float(value[:-1])
        return float(value)
    except ValueError:
        return default


def _call_groq(client: Groq, messages: list[dict[str, str]], max_tokens: int, temperature: float) -> str:
    response = client.chat.completions.create(
        model=settings.groq_model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


async def chat_completion(
    messages: list[dict[str, str]],
    persona: Persona | None = None,
    # Measured against a real full exchange (system prompt + retrieved KB context +
    # experiment context): a thorough answer runs ~1200 completion tokens. 1800
    # comfortably covers that without regularly brushing against Groq's 8000
    # tokens/minute free-tier ceiling — a single ALL-users-shared API key, so this
    # budget is shared app-wide, not per user, and needs real headroom to sustain.
    max_tokens: int = 1800,
    temperature: float = 0.4,
) -> str:
    if not settings.groq_api_key:
        return "GROQ_API_KEY is not set. Add a free Groq API key to your backend .env to enable the AI assistant."

    client = get_client()
    full_messages = [{"role": "system", "content": system_prompt_for(persona)}, *messages]

    for attempt in range(2):
        try:
            return await asyncio.to_thread(_call_groq, client, full_messages, max_tokens, temperature)
        except RateLimitError as exc:
            if attempt == 0:
                # Groq tells us exactly how long until the token bucket refills — wait
                # that long (capped, so a bad header value can't hang a request) via
                # asyncio.sleep, not time.sleep, so this doesn't block every other
                # request the same process is handling while it waits.
                wait_seconds = min(_parse_reset_seconds(exc.response.headers.get("x-ratelimit-reset-tokens")), 10.0)
                await asyncio.sleep(wait_seconds)
                continue
            return "ABBot is getting a lot of questions right now — please wait a few seconds and try again."
        except Exception as exc:  # noqa: BLE001 - surface any other provider error as a chat message
            return f"Sorry, I couldn't reach the AI model right now ({exc})."
    return "ABBot is getting a lot of questions right now — please wait a few seconds and try again."
