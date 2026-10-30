from pathlib import Path

import pytest

from core import chunk_text, extract_chunks


def test_chunk_text_returns_empty_for_blank_input():
    assert chunk_text(" \n ", "demo") == []


def test_chunk_text_keeps_source_metadata_and_overlap():
    text = " ".join(f"word{i}" for i in range(25))
    chunks = chunk_text(text, "TXT file", max_words=10, overlap_words=2)
    assert len(chunks) == 3
    assert chunks[0]["source"] == "TXT file"
    assert chunks[0]["text"].split()[-2:] == chunks[1]["text"].split()[:2]


def test_extract_txt_bytes():
    chunks = extract_chunks("notes.txt", b"Monthly income is 1200 euros.")
    assert len(chunks) == 1
    assert "1200 euros" in chunks[0]["text"]


def test_unsupported_extension_raises():
    with pytest.raises(ValueError):
        extract_chunks("notes.docx", b"some text")


def test_sample_pdf_has_extractable_text():
    project_root = Path(__file__).resolve().parents[1]
    pdf_path = project_root / "sample_data" / "budget.pdf"
    if not pdf_path.exists():
        pytest.skip("Sample PDF is not available")
    chunks = extract_chunks("budget.pdf", pdf_path.read_bytes())
    assert chunks
    assert any("rent" in chunk["text"].lower() for chunk in chunks)
