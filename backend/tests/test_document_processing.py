import pytest

from app.services.document_processing import extract_text, file_extension


def test_file_extension_extracts_lowercase_suffix() -> None:
    assert file_extension("Report.PDF") == "pdf"
    assert file_extension("notes.md") == "md"
    assert file_extension("no_extension") == ""


def test_extract_text_decodes_plain_text() -> None:
    assert extract_text("notes.txt", b"Hello world") == "Hello world"


def test_extract_text_decodes_markdown() -> None:
    assert extract_text("readme.md", b"# Title\n\nBody text.") == "# Title\n\nBody text."


def test_extract_text_rejects_unsupported_extension() -> None:
    with pytest.raises(ValueError, match="Unsupported file type"):
        extract_text("image.png", b"fake bytes")


def test_extract_text_rejects_oversized_file() -> None:
    huge = b"x" * (6 * 1024 * 1024)
    with pytest.raises(ValueError, match="too large"):
        extract_text("notes.txt", huge)


def test_extract_text_rejects_empty_file() -> None:
    with pytest.raises(ValueError, match="empty"):
        extract_text("notes.txt", b"")


def test_extract_text_summarizes_csv_with_stats() -> None:
    csv_bytes = b"group,converted\ncontrol,1\ncontrol,0\ntreatment,1\ntreatment,1\n"
    summary = extract_text("results.csv", csv_bytes)
    assert "4 rows" in summary
    assert "group" in summary
    assert "converted" in summary
    assert "numeric" in summary  # converted is 0/1, treated as numeric
    assert "categorical" in summary  # group is a string column


def test_extract_text_rejects_empty_csv() -> None:
    with pytest.raises(ValueError, match="no rows"):
        extract_text("empty.csv", b"group,converted\n")


def test_extract_text_rejects_corrupt_pdf() -> None:
    with pytest.raises(ValueError, match="Couldn't read this PDF"):
        extract_text("fake.pdf", b"this is not a real pdf")
