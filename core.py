
"""Document loading, chunking, and answer generation utilities."""

from __future__ import annotations

import re
from io import BytesIO
from pathlib import Path
from typing import Any

from pypdf import PdfReader


def chunk_text(
    text: str,
    source: str,
    max_words: int = 160,
    overlap_words: int = 30,
) -> list[dict[str, str]]:
    """Split text into overlapping word chunks."""

    words = text.split()

    if not words:
        return []

    if max_words < 1 or overlap_words < 0 or overlap_words >= max_words:
        raise ValueError("Use max_words > overlap_words >= 0.")

    step = max_words - overlap_words
    chunks = []

    for start in range(0, len(words), step):
        piece = " ".join(words[start:start + max_words]).strip()

        if piece:
            chunks.append({
                "text": piece,
                "source": source,
            })

        if start + max_words >= len(words):
            break

    return chunks


def extract_chunks(
    filename: str,
    file_bytes: bytes,
) -> list[dict[str, str]]:
    """Extract PDF or TXT content into text chunks."""

    suffix = Path(filename).suffix.lower()
    chunks = []

    if suffix == ".pdf":
        reader = PdfReader(BytesIO(file_bytes))

        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""

            chunks.extend(
                chunk_text(page_text, f"PDF page {page_number}")
            )

    elif suffix == ".txt":
        try:
            text = file_bytes.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = file_bytes.decode("latin-1")

        chunks = chunk_text(text, "TXT file")

    else:
        raise ValueError("Please upload a PDF or TXT file.")

    if not chunks:
        raise ValueError(
            "No readable text was found. "
            "Scanned PDFs require OCR, which this version does not include."
        )

    return chunks


def answer_with_gemini(
    question: str,
    retrieved_chunks: list[dict[str, Any]],
    api_key: str,
    model_name: str = "gemini-3.8-flash",
) -> str:
    """Generate an answer grounded in retrieved document passages."""

    from google import genai

    context_parts = []

    for index, item in enumerate(retrieved_chunks, start=1):
        context_parts.append(
            f"[{index}] Source: {item['source']}\n{item['text']}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a careful document question-answering assistant.

Answer the user's question using ONLY the provided document excerpts.

If the excerpts do not contain enough information, say that
the document does not provide enough information.

Do not guess or add outside facts.

Cite factual claims using the excerpt number, for example [1] or [2].

Keep the answer clear and concise.
Answer in the same language as the user's question.

DOCUMENT EXCERPTS:
{context}

USER QUESTION:
{question}
"""

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )

    answer = getattr(response, "text", None)

    if not answer or not answer.strip():
        raise RuntimeError("The AI model returned an empty answer.")

    return answer.strip()


def answer_locally(
    question: str,
    retrieved_chunks: list[dict[str, Any]],
) -> str:
    """Find the most relevant sentence without using an external AI API."""

    stop_words = {
        "what", "is", "the", "a", "an", "my", "your",
        "how", "much", "does", "do", "i", "of", "in",
        "on", "to", "for", "and", "or", "this", "that",
        "me", "user", "please", "tell", "who", "where",
    }

    question_words = {
        word
        for word in re.findall(r"[a-z0-9]+", question.lower())
        if word not in stop_words
    }

    if not question_words:
        return (
            "The document does not provide enough information "
            "to answer this question."
        )

    candidates = []

    for chunk in retrieved_chunks:
        sentences = re.split(
            r"(?<=[.!?])\s+|[●•]\s*",
            chunk["text"],
        )

        for sentence in sentences:
            sentence = sentence.strip()

            if not sentence:
                continue

            sentence_words = set(
                re.findall(r"[a-z0-9]+", sentence.lower())
            )

            overlap = question_words.intersection(sentence_words)

            if overlap:
                score = len(overlap) / len(question_words)

                candidates.append({
                    "score": score,
                    "matches": len(overlap),
                    "sentence": sentence,
                    "source": chunk["source"],
                })

    if not candidates:
        return (
            "The document does not provide enough information "
            "to answer this question."
        )

    candidates.sort(
        key=lambda item: (
            item["matches"],
            item["score"],
            -len(item["sentence"]),
        ),
        reverse=True,
    )

    best = candidates[0]

    if best["score"] < 0.5:
        return (
            "The document does not provide enough information "
            "to answer this question."
        )

    return (
        f"**Local answer (without Gemini):** {best['sentence']}\n\n"
        f"*Source: {best['source']}*"
    )
