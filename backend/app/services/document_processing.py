"""Extracts readable text from an uploaded file so it can be chunked and embedded
the same way the curated knowledge base is. CSV data isn't embedded row-by-row —
raw numbers and category codes don't carry much meaning for similarity search —
it's summarized into prose describing the shape and stats of the data instead,
which is both more useful for retrieval and keeps the LLM's context focused."""

from __future__ import annotations

import io

import pandas as pd
from pypdf import PdfReader

MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024  # 5MB — generous for text/CSV, bounds worst-case latency/memory
SUPPORTED_EXTENSIONS = {"csv", "txt", "md", "pdf"}


def file_extension(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


def extract_text(filename: str, raw_bytes: bytes) -> str:
    ext = file_extension(filename)
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '.{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}.")
    if len(raw_bytes) > MAX_FILE_SIZE_BYTES:
        raise ValueError(f"File is too large ({len(raw_bytes) / 1024 / 1024:.1f}MB). Max size is 5MB.")
    if len(raw_bytes) == 0:
        raise ValueError("File is empty.")

    if ext in ("txt", "md"):
        return raw_bytes.decode("utf-8", errors="replace")
    if ext == "csv":
        return _summarize_csv(raw_bytes)
    return _extract_pdf_text(raw_bytes)


def _summarize_csv(raw_bytes: bytes) -> str:
    try:
        df = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception as exc:
        raise ValueError(f"Couldn't parse this as a CSV: {exc}") from exc
    if df.empty:
        raise ValueError("CSV file has no rows.")

    lines = [f"Dataset with {len(df):,} rows and {len(df.columns)} columns: {', '.join(df.columns)}."]

    for col in df.columns:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            lines.append(
                f"- {col} (numeric): mean={series.mean():.4g}, min={series.min():.4g}, "
                f"max={series.max():.4g}, missing={int(series.isna().sum())}"
            )
        else:
            top_values = series.value_counts().head(5)
            values_desc = ", ".join(f"{v} ({c})" for v, c in top_values.items())
            lines.append(f"- {col} (categorical): {series.nunique()} unique values, top: {values_desc}")

    lines.append("\nFirst 5 rows:\n" + df.head(5).to_string(index=False))
    return "\n".join(lines)


def _extract_pdf_text(raw_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(raw_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:
        raise ValueError(f"Couldn't read this PDF: {exc}") from exc
    text = "\n\n".join(p for p in pages if p.strip())
    if not text.strip():
        raise ValueError("Couldn't extract any text from this PDF — it may be scanned/image-based.")
    return text
