# Implementation Plan — Resume RAG LLM Production Upgrade

This document outlines the audit findings, proposed changes, and verification plan for transitioning the Resume RAG application from a development-prototype status to a production-ready system. We migrate completely from Gemini to OpenRouter, implement offline embeddings and local routing, prevent database pollution, introduce resource caching, and build a high-fidelity recruiter dashboard.

---

## Complete Project Audit

We conducted an audit of the repository contents and local environment configurations. The issues are classified by severity below:

### 🔴 Critical Issues
1. **Broken Gemini SDK Dependencies**: The environment has `llama-index-core` and standard OpenAI components but lacks `llama-index-llms-google-genai` and `llama-index-embeddings-google-genai`. Importing these packages results in immediate `ModuleNotFoundError` crashes.
2. **Missing OpenRouter Service Client**: There is currently no robust OpenRouter service layer in the codebase. All existing code tries to interface directly with Google Gemini SDKs or hardcoded OpenAI paths.
3. **Absence of Secrets and Configurations**: No `OPENROUTER_API_KEY` configurations exist in `src/config.py` or `.env`.

### 🟡 High Issues
1. **Non-Local Routing (Intent Detection)**: The query classifier in `src/classifier.py` uses Gemini LLM completions to determine if a query is single-candidate or multi-candidate. This violates Step 5, which mandates that everything before answer generation (metadata extraction, classification, search, retrieval, routing, ranking, candidate matching, skill extraction) must execute completely locally.
2. **Duplicate Indexing / Collection Pollution**: The indexer in `src/indexer.py` uses `db_client.get_or_create_collection()`. Re-running the indexing process appends duplicate vectors to the collection rather than performing a fresh rebuild, leading to retrieval pollution and skewed similarity scoring.
3. **Cloud Embedding Overhead**: Using `gemini-embedding-001` introduces network latency (~300-800ms) and billing dependencies for indexing and search. We must transition to a fast, local embedding pipeline.

### 🟢 Medium Issues
4. **Lack of Caching**: Heavily initialized resources (Chroma client, HuggingFace embedding models, the index itself, retrievers, OpenRouter client) are initialized repeatedly or loaded ad-hoc, slowing query response times.
5. **Basic User Interface**: The Streamlit interface is minimal and lacks the depth needed for a professional recruiter workflow. It needs diagnostic utilities (debug panel, retrieved context viewer, prompt inspector, similarity score table, performance metrics).
6. **No RAG Evaluation**: The codebase lacks a structured evaluation tool to quantitatively measure the retrieval and generation quality (e.g. faithfulness, context precision).

### 🔵 Low Issues
7. **Basic Print Logging**: Logging is performed via print statements. It must be replaced with Python's standard `logging` library.
8. **Code Quality and Types**: Missing explicit type hints and minor PEP8 formatting discrepancies.

---

## OpenRouter Model Selection

We evaluate several production models available on OpenRouter to determine the best choice for Resume RAG:

| Model ID | Factual Accuracy | Instruction Following | Latency (tokens/s) | Cost per 1M Tokens (Input/Output) | Decision |
|---|---|---|---|---|---|
| **`meta-llama/llama-3.1-70b-instruct`** | **Very High** | **Very High** | **~40** | **$0.52 / $0.75** | **Selected (Default)** |
| `deepseek/deepseek-chat` | Very High | High | ~60 | $0.14 / $0.28 | Recommended Alternative |
| `anthropic/claude-3.5-sonnet` | Excellent (Max) | Excellent (Max) | ~35 | $3.00 / $15.00 | Premium Alternative (Expensive) |
| `qwen/qwen-2.5-72b-instruct` | High | High | ~35 | $0.35 / $0.40 | Good Alternative |

### Selected Model Justification
We choose **`meta-llama/llama-3.1-70b-instruct`** as the default production model. It offers outstanding factual accuracy and strict instruction-following capabilities (essential for preventing hallucinations in recruiter applications) at an extremely low price point. Additionally, we provide the ability to switch models via the `.env` configuration file.

---

## Embedding Model Comparison

We replace `gemini-embedding-001` with a local open-source embedding model. We compare the preferred options below:

| Metric | `gemini-embedding-001` | `BAAI/bge-base-en-v1.5` | `BAAI/bge-large-en-v1.5` |
|---|---|---|---|
| **Location** | Cloud (API) | **Local (CPU/GPU)** | **Local (CPU/GPU)** |
| **Recall / Precision** | Moderate | High | Very High (Slightly better than base) |
| **Latency** | 300 - 800 ms | **10 - 30 ms** | 35 - 90 ms |
| **Embedding Dimensions** | 768 | 768 | 1024 (Heavier storage/index size) |
| **Dependencies** | Network / API Key | None (Offline after download) | None (Offline after download) |
| **Memory footprint** | 0 MB | **~300 MB** | ~1.3 GB |
| **Decision** | Remove | **Selected (Default)** | Supported Alternative |

### Selected Model Justification
We choose **`BAAI/bge-base-en-v1.5`** as the default embedding model. It runs entirely offline, provides exceptional semantic retrieval quality for technical resumes, has a light memory footprint (~300MB), and reduces embedding generation latency to less than 30ms on standard CPUs.

---

## Proposed Changes

We will modify and create the following files to address the requirements:

### `app` Component

#### [NEW] [openrouter_service.py](file:///e:/Resume-Rag-LLM/app/llm/openrouter_service.py)
- Create a production-ready OpenRouter client.
- Implement token/timeout configurations (30-second timeout), a retry mechanism with exponential backoff, detailed exception handling, and a singleton wrapper to avoid redundant client creation.

### `src` Component

#### [MODIFY] [config.py](file:///e:/Resume-Rag-LLM/src/config.py)
- Remove `GEMINI_API_KEY` and all Gemini model strings.
- Add `OPENROUTER_API_KEY`, `OPENROUTER_BASE_URL`, `LLM_MODEL` (default: `meta-llama/llama-3.1-70b-instruct`), and `EMBEDDING_MODEL` (default: `BAAI/bge-base-en-v1.5`).
- Ensure all values are loaded from `.env`.

#### [MODIFY] [classifier.py](file:///e:/Resume-Rag-LLM/src/classifier.py)
- Completely remove Gemini LLM call.
- Implement a robust, rule-based keyword and phrase classifier that runs entirely locally. It identifies:
  - Multi-candidate words (`compare`, `vs`, `both`, `who is better`, `differences`, `similarities`).
  - Aggregations and rankings (`best`, `list all`, `shortlist`, `average`, `who has`).
  - Resolves intents in <1ms without network calls.

#### [MODIFY] [indexer.py](file:///e:/Resume-Rag-LLM/src/indexer.py)
- Replace `GoogleGenAI` and `GoogleGenAIEmbedding` with `OpenAILike` (for OpenRouter) and `HuggingFaceEmbedding` (for offline embeddings).
- Fix the duplicate indexing bug: when rebuilding the index, retrieve the Chroma PersistentClient and call `db_client.delete_collection(COLLECTION_NAME)` explicitly before recreating.
- Verify collection document count equals the resume count (`Collection Count == Resume Count`).

#### [MODIFY] [retriever.py](file:///e:/Resume-Rag-LLM/src/retriever.py)
- Remove Gemini dependencies and update `FilteredQueryEngine` to initialize `HuggingFaceEmbedding` and the OpenRouter client.
- Configure similarity threshold (e.g., 0.5) and top-k retrieval parameters.
- Expose detailed step logs (Query, Intent, Resolved Candidate, Metadata Filters, Similarity Scores, Prompt, Retrieved Nodes) to feed the Streamlit UI Debug Panel.

### Frontend Component

#### [MODIFY] [app.py](file:///e:/Resume-Rag-LLM/app.py)
- Redesign the layout into a premium, recruiter-centric dark-themed dashboard.
- Features to implement:
  - **Recruiter Sidebar**: Quick statistics, candidate list, model info, and system health status.
  - **Interactive Chat UI**: Smooth scrolling bubble view, chat history, quick question buttons.
  - **Timing & Latency Panel**: Displays retrieval time, LLM generation time, and total response time.
  - **Debug Console (Toggleable)**: Inspect intent detection, candidate name matching, active metadata filters, similarity scores, raw prompt, retrieved context nodes, and raw OpenRouter response.
  - **Resume & Context Viewer**: Interactive tabs to view the text content of the retrieved candidate's resume and specific context chunks.
  - **Index Management**: Admin button to "Rebuild Index" showing progress indicators, success feedback, and count verification.

### Dependencies & Tests

#### [MODIFY] [requirements.txt](file:///e:/Resume-Rag-LLM/requirements.txt)
- Remove Google GenAI dependencies.
- Add `llama-index-embeddings-huggingface` and `sentence-transformers` for local embedding support.

#### [NEW] [evaluator.py](file:///e:/Resume-Rag-LLM/src/evaluator.py)
- Create a RAG evaluation script that computes key RAG parameters:
  - Faithfulness
  - Context Recall
  - Context Precision
  - Answer Relevancy
- Generates detailed local JSON reports for recruiter audit purposes.

#### [MODIFY] [test_rag.py](file:///e:/Resume-Rag-LLM/tests/test_rag.py)
- Update mock frameworks to mock the OpenRouter service instead of Gemini.
- Adjust tests to verify local rule-based intent routing.

#### [MODIFY] [test_retrieval.py](file:///e:/Resume-Rag-LLM/tests/test_retrieval.py)
- Adapt integration tests to cover the local embedding model and OpenRouter service calls.

---

## Verification Plan

### Automated Tests
We will verify codebase stability using the existing and updated test suites. Run:
```bash
# Set dummy credentials for testing local modules
$env:OPENROUTER_API_KEY="sk-or-test-key-here"

# Run tests using the virtual environment interpreter
.venv\Scripts\python -m unittest tests/test_rag.py
.venv\Scripts\python -m unittest tests/test_retrieval.py
```

### Manual Verification
1. Launch the dashboard: `.venv\Scripts\streamlit run app.py`
2. Select **Index Resumes** to build a clean vector store; verify "Collection Count == Resume Count" output.
3. Submit the following test queries and verify outcomes:
   - **"Summarize Trinadh's resume"** (Verify summary details match Trinadh Kumar Reddi).
   - **"What are Trinadh's certifications?"** (Verify listed certifications).
   - **"Compare Trinadh and Rahul"** (Verify query is blocked as a comparison query).
   - **"Recommend the best AI Engineer"** (Verify query is blocked as a ranking/recommendation query).
   - **"Find candidates with LangGraph"** (Verify query is blocked as an aggregation/search query).
4. Verify in the Debug Panel:
   - OpenRouter is called exactly once.
   - Metadata filter uses `{"candidate_name": "Trinadh Kumar Reddi"}`.
   - Retrieval and LLM generation latencies are tracked and displayed.

### Git Commit History (Step 14 Outline)
We will make the following commits:
1. `git add .` -> `git commit -m "Replace Gemini with OpenRouter integration"`
2. `git add .` -> `git commit -m "Upgrade embedding model and improve semantic retrieval"`
3. `git add .` -> `git commit -m "Fix duplicate indexing and optimize ChromaDB"`
4. `git add .` -> `git commit -m "Improve recruiter dashboard and Streamlit UI"`
5. `git add .` -> `git commit -m "Add evaluation, logging and performance monitoring"`
