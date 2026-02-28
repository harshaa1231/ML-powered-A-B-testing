from __future__ import annotations

import os
import requests
from typing import List, Dict, Optional

HF_API_URL = "https://router.huggingface.co/v1/chat/completions"
DEFAULT_MODEL = "meta-llama/Llama-3.2-3B-Instruct"

SYSTEM_PROMPT = """You are an expert A/B testing and data science advisor named ABBot, built into AB Testing Pro — a platform for running and understanding A/B experiments.

You help users with:
- Understanding A/B test results in plain English
- Explaining statistics (p-values, confidence, effect size, uplift) without jargon
- Giving business advice based on experiment outcomes
- Explaining machine learning concepts used in the platform
- Answering general data science and experimentation questions
- Helping users decide what to test next

Guidelines:
- Always be clear, friendly, and approachable — explain things like a knowledgeable colleague, not a textbook
- Use concrete examples and analogies to explain complex concepts
- When given test results (p-value, uplift, sample size), give specific advice for those numbers
- Keep answers focused and practical — what does this mean and what should the user DO?
- If asked about something outside A/B testing or data science, gently redirect back to your expertise
- Never use unnecessary jargon. If you must use a technical term, immediately explain it.
"""


def chat_with_hf(
    messages: List[Dict[str, str]],
    model: str = DEFAULT_MODEL,
    max_tokens: int = 600,
    temperature: float = 0.7,
) -> str:
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        return "HF_TOKEN environment variable not set. Please add your Hugging Face token to run the AI chat."

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages

    payload = {
        "model": model,
        "messages": full_messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    try:
        r = requests.post(HF_API_URL, headers=headers, json=payload, timeout=45)
        if r.status_code == 200:
            data = r.json()
            return data["choices"][0]["message"]["content"].strip()
        elif r.status_code == 401:
            return "Invalid Hugging Face token. Please check your HF_TOKEN secret."
        elif r.status_code == 429:
            return "Rate limit reached on Hugging Face. Please wait a moment and try again."
        elif r.status_code == 503:
            return "The AI model is currently loading — this can take 20-30 seconds on first use. Please try again in a moment."
        else:
            return f"Sorry, I couldn't get a response right now (error {r.status_code}). Please try again."
    except requests.Timeout:
        return "The request timed out — the model may be loading. Please try again in a few seconds."
    except Exception as e:
        return f"Something went wrong: {str(e)}"


def build_context_message(test_results: Optional[Dict] = None) -> str:
    if not test_results:
        return ""
    p = test_results.get("p_value", None)
    uplift = test_results.get("uplift_percentage", None)
    sig = test_results.get("is_significant", None)
    n_c = test_results.get("n_control", None)
    n_t = test_results.get("n_treatment", None)
    domain = test_results.get("domain", "general")
    test_name = test_results.get("test_name", "")

    parts = ["The user has just run an A/B test with the following results:"]
    if test_name:
        parts.append(f"- Test type: {test_name}")
    if p is not None:
        parts.append(f"- P-value: {p:.4f}")
    if sig is not None:
        parts.append(f"- Statistically significant: {'Yes' if sig else 'No'}")
    if uplift is not None:
        parts.append(f"- Uplift (B vs A): {uplift:.2f}%")
    if n_c and n_t:
        parts.append(f"- Sample sizes: {n_c:,} in control, {n_t:,} in treatment")
    if domain != "general":
        parts.append(f"- Domain: {domain}")
    parts.append("Please tailor your responses to these specific results.")
    return "\n".join(parts)


STARTER_QUESTIONS = [
    "What is A/B testing and why does it matter?",
    "How do I know if my sample size is large enough?",
    "What does a p-value actually mean in plain English?",
    "My test is inconclusive — what should I do next?",
    "What's the difference between statistical and practical significance?",
    "How do I explain A/B test results to my manager?",
    "What makes a good A/B test hypothesis?",
    "How long should I run my A/B test?",
]
