from __future__ import annotations

import os
import requests
from typing import List, Dict, Optional

DEFAULT_MODEL = "microsoft/DialoGPT-medium"

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

    # Use the free Inference API endpoint
    API_URL = f"https://api-inference.huggingface.co/models/{model}"
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    # Convert messages to prompt format for free models
    conversation = ""
    for msg in messages:
        if msg["role"] == "user":
            conversation += f"User: {msg['content']}\n"
        elif msg["role"] == "assistant":
            conversation += f"Assistant: {msg['content']}\n"
    
    prompt = f"{SYSTEM_PROMPT}\n\n{conversation}\nAssistant:"

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "return_full_text": False,
            "do_sample": True,
        }
    }

    try:
        r = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list) and len(data) > 0:
                return data[0].get("generated_text", "I understand, but I'm having trouble formulating a complete response about A/B testing right now.").strip()
            else:
                return str(data)
        elif r.status_code == 503:
            return "The model is loading (this is normal for free models). Please try again in 10-15 seconds."
        else:
            # Debug: Print the actual error
            print(f"Debug - Error {r.status_code}: {r.text[:300]}")
            return f"Sorry, I couldn't get a response right now (error {r.status_code}). Please try again."
    except requests.Timeout:
        return "The request timed out — the free model may be loading. Please try again in a few seconds."
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
