# AI Document Assistant

A small Retrieval-Augmented Generation (RAG) demo that lets a user ask questions about a PDF or TXT document. It uses local semantic search to retrieve relevant passages and can optionally use the Gemini API to generate an answer grounded in those passages.

## Features

- Streamlit web UI with document upload and chat-style questions.
- PDF text extraction with `pypdf` and TXT support.
- Overlapping text chunking with source metadata (PDF page or TXT file).
- Local semantic retrieval using `sentence-transformers/all-MiniLM-L6-v2`.
- Optional Gemini-generated answers with source markers; retrieval still works without an API key.
- Error handling and unit tests for document processing.

## Requirements

- Python 3.10 or newer recommended.
- Internet access the first time the embedding model is downloaded.
- A Gemini API key is optional. Without one, the app uses semantic retrieval and displays the most relevant passages. A key enables generated answers.

## Run locally on Linux

Open a terminal in this project folder and run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

Streamlit prints a local URL (usually `http://localhost:8501`) to open in your browser.

## Enable generated answers (optional)

1. Open [Google AI Studio](https://aistudio.google.com/apikey) and create an API key for the Gemini API.
2. Copy `.env.example` to a new file named `.env` in this folder.
3. Add the key on the `GEMINI_API_KEY=` line. Do not share the key or commit `.env` to GitHub.
4. Restart Streamlit. The sidebar should indicate that generative AI is enabled.

`GEMINI_MODEL` can be changed in `.env`; the default in this project is `gemini-3.8-flash`. Availability and free-tier limits depend on the account and model. Review Google's current API pricing and terms before relying on a free tier.

**Privacy note:** do not use confidential or sensitive documents for this demo. Google's Gemini API free tier may use submitted content to improve products. See Google's [Gemini API pricing and data-use information](https://ai.google.dev/gemini-api/docs/pricing).

## Use the built-in demo

Start the app and choose **Use example budget PDF** in the sidebar. Try questions such as:

- `How much do I pay for housing?`
- `How much do I spend on groceries?`
- `What is my monthly income?`

To use your own document, choose **Upload a PDF or TXT** in the sidebar.

## Run tests

With the virtual environment active:

```bash
pytest -q
```

## How it works

1. Extract text from the uploaded document.
2. Split it into overlapping passages while keeping source labels.
3. Encode the passages with a local sentence-transformer embedding model.
4. Encode the user's question and retrieve the most semantically similar passages.
5. If a Gemini API key is configured, send only the retrieved passages and question to Gemini with an instruction to answer from the evidence and cite source numbers. Otherwise, show the passages directly.

## Limitations

- Scanned/image-only PDFs are not supported because OCR is not included.
- Semantic similarity is a retrieval signal, not a confidence probability; inspect the sources to verify important answers.
- The project is an educational prototype, not a production document system. It does not provide authentication, persistent storage, or enterprise privacy controls.

## CV wording (only after testing)

**AI Document Assistant (Python, Streamlit, RAG, Sentence Transformers, Gemini API)** — Built a document Q&A prototype with PDF/TXT ingestion, overlapping chunking, local semantic retrieval, source-aware context assembly, and optional LLM answer generation with citations.
