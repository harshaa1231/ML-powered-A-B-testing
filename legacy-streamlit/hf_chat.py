from __future__ import annotations

import os
import io
import requests
import base64
import pandas as pd
from typing import List, Dict, Optional

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are an expert A/B testing and data science advisor named ABBot, built into AB Testing Pro — a platform for running and understanding A/B experiments.

You help users with:
- Understanding A/B test results in plain English
- Explaining statistics (p-values, confidence, effect size, uplift) without jargon
- Giving business advice based on experiment outcomes
- Explaining machine learning concepts used in the platform
- Answering general data science and experimentation questions
- Helping users decide what to test next
- Analyzing uploaded data files (CSV, Excel, etc.) and providing insights

Guidelines:
- Always be clear, friendly, and approachable — explain things like a knowledgeable colleague, not a textbook
- Use concrete examples and analogies to explain complex concepts
- When given test results (p-value, uplift, sample size), give specific advice for those numbers
- When given data from an uploaded file, analyze it thoroughly — summarize key stats, identify patterns, and give actionable recommendations
- Keep answers focused and practical — what does this mean and what should the user DO?
- If asked about something outside A/B testing or data science, gently redirect back to your expertise
- Never use unnecessary jargon. If you must use a technical term, immediately explain it.
"""

# ==============================================================================
# FILE PARSING
# ==============================================================================

def parse_uploaded_file(uploaded_file) -> Dict:
    """
    Parse an uploaded file and return a dict with:
      - 'type': file type (csv, excel, pdf, image, text)
      - 'summary': text summary of the file content for LLM context
      - 'dataframe': pandas DataFrame if applicable (csv/excel)
      - 'error': error message if parsing failed
    """
    filename = uploaded_file.name.lower()
    result = {'type': None, 'summary': '', 'dataframe': None, 'error': None}

    try:
        if filename.endswith('.csv'):
            result['type'] = 'csv'
            df = pd.read_csv(uploaded_file)
            result['dataframe'] = df
            result['summary'] = _summarize_dataframe(df, filename)

        elif filename.endswith(('.xlsx', '.xls')):
            result['type'] = 'excel'
            df = pd.read_excel(uploaded_file)
            result['dataframe'] = df
            result['summary'] = _summarize_dataframe(df, filename)

        elif filename.endswith('.pdf'):
            result['type'] = 'pdf'
            result['summary'] = _parse_pdf(uploaded_file)

        elif filename.endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp')):
            result['type'] = 'image'
            result['summary'] = "[User uploaded an image. I cannot directly analyze images, but I can help if you describe what's in the image or paste the data from it.]"

        elif filename.endswith(('.txt', '.md', '.log')):
            result['type'] = 'text'
            content = uploaded_file.read().decode('utf-8', errors='ignore')
            # Truncate if too long
            if len(content) > 8000:
                content = content[:8000] + "\n\n[... file truncated for length ...]"
            result['summary'] = f"Content of {uploaded_file.name}:\n\n{content}"

        elif filename.endswith('.json'):
            result['type'] = 'json'
            import json
            content = json.loads(uploaded_file.read().decode('utf-8'))
            # Try to make a DataFrame from it
            try:
                df = pd.json_normalize(content) if isinstance(content, (list, dict)) else None
                if df is not None and not df.empty:
                    result['dataframe'] = df
                    result['summary'] = _summarize_dataframe(df, filename)
                else:
                    result['summary'] = f"JSON file content:\n{json.dumps(content, indent=2)[:5000]}"
            except Exception:
                result['summary'] = f"JSON file content:\n{json.dumps(content, indent=2)[:5000]}"

        else:
            result['error'] = f"Unsupported file type: {filename.split('.')[-1]}. Supported: CSV, Excel, PDF, TXT, JSON, images."

    except Exception as e:
        result['error'] = f"Error reading file: {str(e)}"

    return result


def _summarize_dataframe(df: pd.DataFrame, filename: str) -> str:
    """Create a comprehensive text summary of a DataFrame for LLM context."""
    lines = []
    lines.append(f"=== Uploaded File: {filename} ===")
    lines.append(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    lines.append("")

    # Column info
    lines.append("Columns:")
    for col in df.columns:
        dtype = str(df[col].dtype)
        nunique = df[col].nunique()
        null_count = df[col].isnull().sum()
        sample_vals = df[col].dropna().head(3).tolist()
        lines.append(f"  - {col} ({dtype}): {nunique} unique values, {null_count} nulls, sample: {sample_vals}")
    lines.append("")

    # Basic stats for numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        lines.append("Numeric Summary:")
        desc = df[numeric_cols].describe().round(4)
        lines.append(desc.to_string())
        lines.append("")

    # Categorical column value counts (top 5)
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols[:5]:  # Limit to 5 categorical columns
        lines.append(f"Value counts for '{col}':")
        vc = df[col].value_counts().head(5)
        for val, count in vc.items():
            lines.append(f"  {val}: {count} ({count/len(df)*100:.1f}%)")
        lines.append("")

    # First few rows
    lines.append("First 5 rows:")
    lines.append(df.head(5).to_string(index=False))

    # Truncate if too long
    full_text = "\n".join(lines)
    if len(full_text) > 6000:
        full_text = full_text[:6000] + "\n\n[... summary truncated for length ...]"

    return full_text


def _parse_pdf(uploaded_file) -> str:
    """Try to extract text from a PDF file."""
    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(uploaded_file.read()))
        text = ""
        for page in reader.pages[:20]:  # Limit to 20 pages
            text += page.extract_text() or ""
        if not text.strip():
            return "[PDF uploaded but no text could be extracted. It may be a scanned document. Please copy-paste the relevant data instead.]"
        if len(text) > 8000:
            text = text[:8000] + "\n\n[... PDF truncated for length ...]"
        return f"Content of uploaded PDF:\n\n{text}"
    except ImportError:
        return "[PDF uploaded but the PDF reader library (pypdf) is not installed. Please upload a CSV or Excel file instead, or copy-paste the relevant data.]"
    except Exception as e:
        return f"[Error reading PDF: {str(e)}. Please try uploading a CSV or Excel file instead.]"


# ==============================================================================
# LLM CHAT
# ==============================================================================

def chat_with_hf(
    messages: List[Dict[str, str]],
    model: str = DEFAULT_MODEL,
    max_tokens: int = 1200,
    temperature: float = 0.7,
) -> str:
    """Send messages to Groq API and return the response."""
    token = os.environ.get("GROQ_API_KEY", "")
    if not token:
        return "GROQ_API_KEY not set. Please add your Groq API key to Streamlit secrets (Settings → Secrets → add GROQ_API_KEY = \"your-key\")."

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
        r = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
        if r.status_code == 200:
            data = r.json()
            return data["choices"][0]["message"]["content"].strip()
        elif r.status_code == 401:
            return "Invalid Groq API key. Please check your GROQ_API_KEY in Streamlit secrets."
        elif r.status_code == 429:
            return "Rate limit reached. Please wait a moment and try again."
        else:
            return f"Sorry, I couldn't get a response right now (error {r.status_code}). Please try again."
    except requests.Timeout:
        return "The request timed out. Please try again."
    except Exception as e:
        return f"Something went wrong: {str(e)}"


# ==============================================================================
# CONTEXT BUILDING
# ==============================================================================

def build_context_message(test_results: Optional[Dict] = None) -> str:
    """Build context from test results to inject into chat."""
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


def build_file_context_message(file_summary: str) -> str:
    """Build context from an uploaded file summary."""
    return f"The user has uploaded a file. Here is a summary of its contents:\n\n{file_summary}\n\nPlease analyze this data and provide insights. If it looks like A/B test data, identify the groups and metrics and suggest what tests to run."


# ==============================================================================
# STARTER QUESTIONS
# ==============================================================================

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
