
"""Streamlit interface for the AI Document Assistant."""

import hashlib
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util

from core import (
    answer_locally,
    answer_with_gemini,
    extract_chunks,
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
SAMPLE_PDF = BASE_DIR / "sample_data" / "budget.pdf"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

load_dotenv(BASE_DIR / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")


# --------------------------------------------------
# Page settings
# --------------------------------------------------

st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📄",
    layout="centered",
)

st.title("📄 AI Document Assistant")

st.write(
    "Upload a PDF or TXT document and ask questions about its content."
)


# --------------------------------------------------
# Load embedding model
# --------------------------------------------------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


# --------------------------------------------------
# Semantic retrieval
# --------------------------------------------------

def retrieve(
    question,
    chunks,
    embeddings,
    model,
    top_k=3,
):
    """Retrieve the most relevant document chunks."""

    question_embedding = model.encode(
        question,
        convert_to_tensor=True,
    )

    similarities = util.cos_sim(
        question_embedding,
        embeddings,
    )[0]

    number_of_results = min(top_k, len(chunks))

    top_results = similarities.topk(number_of_results)

    results = []

    for score, index in zip(
        top_results.values,
        top_results.indices,
    ):
        chunk = chunks[int(index)]

        results.append({
            "text": chunk["text"],
            "source": chunk["source"],
            "similarity": float(score),
        })

    return results


# --------------------------------------------------
# Document selection
# --------------------------------------------------

st.sidebar.header("Document settings")

document_option = st.sidebar.radio(
    "Choose your document:",
    [
        "Use sample budget PDF",
        "Upload my own document",
    ],
)

filename = None
file_bytes = None

if document_option == "Use sample budget PDF":

    if SAMPLE_PDF.exists():
        filename = SAMPLE_PDF.name
        file_bytes = SAMPLE_PDF.read_bytes()
    else:
        st.error("The sample PDF was not found.")
        st.stop()

else:

    uploaded_file = st.sidebar.file_uploader(
        "Upload a PDF or TXT file",
        type=["pdf", "txt"],
    )

    if uploaded_file is None:
        st.info("Please upload a PDF or TXT document.")
        st.stop()

    filename = uploaded_file.name
    file_bytes = uploaded_file.getvalue()


# --------------------------------------------------
# Extract and index document
# --------------------------------------------------

document_hash = hashlib.sha256(
    filename.encode("utf-8") + file_bytes
).hexdigest()

if st.session_state.get("document_hash") != document_hash:

    try:
        with st.spinner("Reading your document..."):

            chunks = extract_chunks(filename, file_bytes)

            embedding_model = load_embedding_model()

            embeddings = embedding_model.encode(
                [chunk["text"] for chunk in chunks],
                convert_to_tensor=True,
            )

            st.session_state.document_hash = document_hash
            st.session_state.chunks = chunks
            st.session_state.embeddings = embeddings
            st.session_state.messages = []

    except Exception as exc:
        st.error(f"Could not process the document: {exc}")
        st.stop()


chunks = st.session_state.chunks
embeddings = st.session_state.embeddings

embedding_model = load_embedding_model()


# --------------------------------------------------
# Document information
# --------------------------------------------------

st.success(f"Document ready: {filename}")

st.caption(f"Number of text chunks: {len(chunks)}")

max_chunks = min(5, len(chunks))

if max_chunks >= 2:
    top_k = st.sidebar.slider(
        "Number of passages to retrieve",
        min_value=1,
        max_value=max_chunks,
        value=min(3, max_chunks),
    )
else:
    top_k = 1


# --------------------------------------------------
# Chat history
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("Retrieved source passages"):
                for source in message["sources"]:
                    st.markdown(
                        f"**{source['source']}** · "
                        f"similarity {source['similarity']:.3f}\n\n"
                        f"> {source['text']}"
                    )


# --------------------------------------------------
# User question and answer
# --------------------------------------------------

question = st.chat_input(
    "Ask a question about your document"
)

if question:

    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Searching the document..."):

            try:
                results = retrieve(
                    question,
                    chunks,
                    embeddings,
                    embedding_model,
                    top_k,
                )

                if GEMINI_API_KEY:

                    try:
                        answer = answer_with_gemini(
                            question,
                            results,
                            GEMINI_API_KEY,
                            GEMINI_MODEL,
                        )

                    except Exception:
                        answer = (
                            "Gemini is temporarily unavailable. "
                            "I will use local retrieval instead.\n\n"
                            + answer_locally(question, results)
                        )

                else:
                    answer = answer_locally(question, results)

                st.markdown(answer)

                with st.expander("Retrieved source passages"):

                    for source in results:
                        st.markdown(
                            f"**{source['source']}** · "
                            f"similarity {source['similarity']:.3f}\n\n"
                            f"> {source['text']}"
                        )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": results,
                })

            except Exception as exc:

                error_text = (
                    f"An unexpected error occurred: {exc}"
                )

                st.error(error_text)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_text,
                })
