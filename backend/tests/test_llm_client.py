"""Fast, no-network unit tests for persona-aware prompt selection — the actual
Groq call is exercised by the live integration tests (test_api_chat.py,
test_api_practice.py, test_api_analytics.py); this just verifies the two
personas genuinely get different instructions, not the same prompt twice."""

from app.rag.llm_client import BUSINESS_PROMPT, LEARNER_PROMPT, system_prompt_for


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
