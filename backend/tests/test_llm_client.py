"""Fast, no-network unit tests for persona-aware prompt selection and rate-limit
resilience — the actual Groq call is otherwise exercised by the live integration
tests (test_api_chat.py, test_api_practice.py, test_api_analytics.py); the prompt
tests just verify the two personas genuinely get different instructions, and the
rate-limit tests verify a real, previously-absent failure mode: Groq's free tier
has a single, app-wide 8000-tokens/minute ceiling shared by every user through one
API key, and a request that lands right when it's exhausted used to either crash
or need the caller to handle it — now it retries once using Groq's own reported
reset time, then degrades to a clear message instead of a raw exception dump."""

import httpx
import pytest
from groq import RateLimitError

from app.rag import llm_client
from app.rag.llm_client import BUSINESS_PROMPT, LEARNER_PROMPT, chat_completion, system_prompt_for


def _rate_limit_error(reset_header: str = "10ms") -> RateLimitError:
    response = httpx.Response(
        status_code=429,
        headers={"x-ratelimit-reset-tokens": reset_header},
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
    )
    return RateLimitError("rate limited", response=response, body=None)


def test_business_and_learner_prompts_are_different() -> None:
    assert system_prompt_for("business") != system_prompt_for("learner")


def test_business_persona_gets_business_prompt() -> None:
    assert system_prompt_for("business") == BUSINESS_PROMPT
    assert "decisions, not theory" in system_prompt_for("business")


def test_learner_persona_gets_learner_prompt() -> None:
    assert system_prompt_for("learner") == LEARNER_PROMPT
    assert "first principles" in system_prompt_for("learner")


def test_unknown_or_missing_persona_falls_back_to_business() -> None:
    assert system_prompt_for(None) == BUSINESS_PROMPT


async def test_chat_completion_retries_once_after_rate_limit_then_succeeds(monkeypatch) -> None:
    monkeypatch.setattr(llm_client.settings, "groq_api_key", "fake-key-for-test")
    call_count = {"n": 0}

    def fake_call_groq(client, messages, max_tokens, temperature):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise _rate_limit_error()
        return "real answer"

    monkeypatch.setattr(llm_client, "_call_groq", fake_call_groq)

    result = await chat_completion([{"role": "user", "content": "hi"}])
    assert result == "real answer"
    assert call_count["n"] == 2


async def test_chat_completion_gives_a_friendly_message_after_repeated_rate_limits(monkeypatch) -> None:
    monkeypatch.setattr(llm_client.settings, "groq_api_key", "fake-key-for-test")
    monkeypatch.setattr(llm_client, "_call_groq", lambda *a, **kw: (_ for _ in ()).throw(_rate_limit_error()))

    result = await chat_completion([{"role": "user", "content": "hi"}])
    assert "getting a lot of questions" in result.lower()


async def test_chat_completion_returns_setup_message_without_api_key(monkeypatch) -> None:
    monkeypatch.setattr(llm_client.settings, "groq_api_key", "")
    result = await chat_completion([{"role": "user", "content": "hi"}])
    assert "GROQ_API_KEY" in result


def test_parse_reset_seconds_handles_milliseconds_and_seconds() -> None:
    assert llm_client._parse_reset_seconds("742ms") == pytest.approx(0.742)
    assert llm_client._parse_reset_seconds("21.57s") == pytest.approx(21.57)
    assert llm_client._parse_reset_seconds(None) == 2.0
    assert llm_client._parse_reset_seconds("garbage") == 2.0
