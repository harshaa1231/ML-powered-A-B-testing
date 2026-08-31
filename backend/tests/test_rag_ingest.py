from app.rag.chunking import chunk_text as _chunk_text
from app.rag.ingest import _parse_title


def test_parse_title_extracts_h1() -> None:
    markdown = "# What is A/B Testing?\n\nSome content here."
    assert _parse_title(markdown, fallback="fallback") == "What is A/B Testing?"


def test_parse_title_falls_back_when_no_heading() -> None:
    assert _parse_title("no heading here", fallback="my-slug") == "my-slug"


def test_chunk_text_splits_long_documents() -> None:
    paragraphs = [f"Paragraph {i} " + ("word " * 50) for i in range(10)]
    text = "\n\n".join(paragraphs)

    chunks = _chunk_text(text, chunk_size=400, overlap=0)

    assert len(chunks) > 1
    assert all(len(c) <= 500 for c in chunks)  # some slack for paragraph boundaries


def test_chunk_text_keeps_short_document_as_one_chunk() -> None:
    text = "Short paragraph one.\n\nShort paragraph two."
    chunks = _chunk_text(text, chunk_size=800, overlap=100)
    assert len(chunks) == 1
    assert "Short paragraph one." in chunks[0]
    assert "Short paragraph two." in chunks[0]
