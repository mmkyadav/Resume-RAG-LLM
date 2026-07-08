# Implementation Plan — Resume RAG LLM

## Overview

A Retrieval-Augmented Generation (RAG) pipeline that answers natural-language questions about individual candidates from a directory of resumes. The pipeline includes smart query routing, fuzzy name resolution, multi-resume blocking, and a premium Streamlit UI.

---

## Architecture

```
app.py  (Streamlit UI)
   │  query(str)  +  Gemini API key (from sidebar)
   ▼
src/retriever.py  (FilteredQueryEngine)
   ├── src/classifier.py   →  Gemini LLM or local heuristic
   ├── src/matcher.py      →  rapidfuzz fuzzy name matching
   └── LlamaIndex  ──►  Chroma DB  (filtered by candidate_name metadata)
```

---

## Tech Stack

| Component | Technology | Notes |
|---|---|---|
| **Frontend** | Streamlit | Dark glassmorphism UI |
| **Orchestration** | LlamaIndex ≥ 0.10 | VectorStoreIndex + MetadataFilters |
| **LLM** | `models/gemini-1.5-flash` | Google Gemini via `llama-index-llms-gemini` |
| **Embeddings** | `models/text-embedding-004` | Google Gemini via `llama-index-embeddings-gemini` |
| **Vector DB** | Chroma DB | Local persistent storage in `chroma_db/` |
| **PDF Parser** | pypdf | |
| **DOCX Parser** | python-docx | |
| **Fuzzy Matching** | rapidfuzz | Token-based candidate name resolution |

---

## Person 1 — Ingestion

**Branch:** `dev-person-1-ingestion`

### Deliverables

| File | Description |
|---|---|
| `src/config.py` | Loads `GEMINI_API_KEY`, sets default LLM/embedding model paths |
| `src/parser.py` | Extracts candidate names from filenames and parses PDF/DOCX content |
| `src/indexer.py` | Builds/refreshes Chroma DB collection using Gemini embeddings |

### Key decisions
- Candidate name is extracted from the filename suffix: `<Description> - <Candidate Full Name>.pdf`
- Each resume is stored as a single LlamaIndex `Document` with `candidate_name` metadata for precise filter retrieval.
- `setup_llamaindex_settings(api_key=None)` accepts an optional key so the frontend can supply the key at runtime.

---

## Person 2 — Fuzzy Name Matching & Query Routing

**Branch:** `dev-person-2-retrieval`

### Deliverables

| File | Description |
|---|---|
| `src/matcher.py` | Token-based fuzzy name matcher; resolves typos and shortforms (e.g. "pawan" → "pavanteja kamma") |
| `src/classifier.py` | Gemini LLM-based query classifier (single vs. multi-candidate); includes local keyword heuristic fallback |
| `src/retriever.py` | `FilteredQueryEngine` — orchestrates classification → name resolution → metadata-filtered LlamaIndex query |

### Key decisions
- Classifier uses `Gemini(model="models/gemini-1.5-flash")` via `llama-index-llms-gemini`.
- Both `QueryClassifier` and `FilteredQueryEngine` accept an optional `api_key` argument injected at runtime by the Streamlit sidebar.
- If API key is missing, mock mode is enabled with a clear user-facing message.

---

## Person 3 — Frontend & Integration

**Branch:** `dev-person-3-frontend`

### Deliverables

| File | Description |
|---|---|
| `app.py` | Premium Streamlit UI with chat history, stats, spelling alerts, blocked query alerts |
| `tests/test_rag.py` | 53 unit tests for name parsing, fuzzy correction, comparison blocking, and single-candidate QA |
| `README.md` | Full setup, usage, and troubleshooting documentation |

### Key UI features
- **🔑 Gemini API Key input** in the sidebar: paste your key at runtime without editing `.env`.
- Engine automatically re-initialises when the key changes — status indicator updates to `✅ Pipeline Ready (Live LLM)` or `(Mock Mode)`.
- Chat window uses `st.container(height=500)` for a native scrollable layout (no blank-block artefact).
- Text input box styled with white background and black text for clear visibility.

---

## Environment Configuration

Create a `.env` file in the project root:

```dotenv
GEMINI_API_KEY=AIzaSy_your_key_here

# Optional model overrides:
# LLM_MODEL=models/gemini-1.5-pro
# EMBEDDING_MODEL=models/text-embedding-004
```

> **Alternative:** You can skip `.env` entirely and paste your API key directly into the **🔑 Gemini API Key** sidebar field in the Streamlit app.

---

## Setup Steps

1. `pip install -r requirements.txt`
2. Set `GEMINI_API_KEY` in `.env` **or** use the sidebar input
3. Place PDF/DOCX resumes in `Resumes/`
4. `python src/indexer.py` (one-time index build, requires Gemini key)
5. `streamlit run app.py`

---

## Git Commit Log (Key Changes)

| Commit | Message |
|---|---|
| `feat: implement fuzzy name resolution with spelling suggestions` | Person 2 — matcher.py |
| `feat: implement LLM-based query classifier (single vs multi-resume)` | Person 2 — classifier.py |
| `feat: implement filtered retrieval query engine` | Person 2 — retriever.py |
| `fix: text input styling and remove empty chat window block` | Person 3 — app.py styling |
| `feat: migrate backend to Google Gemini LLM and Embeddings (GEMINI_API_KEY)` | All backend src/ files |
| `feat: add Gemini API key input in sidebar with live engine re-init` | app.py sidebar |
| `docs: update README and implementation_plan for Gemini migration` | README.md + this file |

---

## Verification

### Automated Tests

```bash
# Full suite (53 tests)
python -m unittest tests/test_rag.py

# Retrieval pipeline tests
python -m unittest tests/test_retrieval.py
```

### Manual Verification

1. Run `streamlit run app.py`
2. Paste your `GEMINI_API_KEY` in the sidebar
3. Verify status badge shows **✅ Pipeline Ready (Live LLM)**
4. Run a sample query: `"What are Ashok's Python skills?"`
5. Verify spelling correction fires for: `"Tell me about pawan's experience"`
6. Verify comparison blocking for: `"Compare Ashok and Pawan"`
