# Resume RAG LLM 🤖

A **Retrieval-Augmented Generation (RAG)** system that answers detailed questions about individual candidates from a directory of resumes. Built with **LlamaIndex**, **Chroma DB**, **Meta Llama 3.1 LLM (via OpenRouter)**, and a premium **Streamlit** interface.

---

## Features

| Feature | Description |
|---|---|
| 🔍 **Single-Candidate QA** | Deep-dive into any candidate's skills, experience, education, or projects |
| ✏️ **Spelling Correction** | Fuzzy matching maps inputs like "pawan" → "pavanteja kamma" and "yasasvi" → "yasaswi kotha" |
| 🚫 **Multi-Resume Blocking** | Comparison, ranking, and aggregation queries are automatically rejected to enforce routing safety |
| 💬 **Chat History** | Full conversation history with per-message metadata |
| 📊 **Live Stats** | Query counts, latency metrics, spelling correction alerts, and blocked-query counters |
| 🎨 **Premium Dark UI** | Sleek Glassmorphism dark mode with professional Outfit & Space Mono typography |
| 🔍 **Debug Diagnostics** | Toggleable details showing exact retrieval scores, intent logs, and processing timings |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    app.py  (Streamlit UI)                   │
│   Chat window · Stats · Spelling alerts · Blocked alerts    │
│   🔍 Collapsible Debug Panel & timing diagnostics           │
└────────────────────────┬────────────────────────────────────┘
                         │ query(str)
┌────────────────────────▼────────────────────────────────────┐
│              src/retriever.py  (FilteredQueryEngine)        │
│  1. QueryClassifier  →  block comparison/multi queries      │
│  2. FuzzyNameMatcher →  resolve candidate name (+ typos)    │
│  3. LlamaIndex + ChromaDB  →  filtered semantic retrieval   │
└─────────────────────────────────────────────────────────────┘
              │                    │                    │
   src/classifier.py      src/matcher.py        src/indexer.py
   Heuristic fallback     rapidfuzz             Build Chroma index
   classifier             fuzzy matching         with Local BGE Embeddings
```

---

## Tech Stack

| Component | Technology |
|---|---|
| **Frontend** | [Streamlit](https://streamlit.io/) |
| **Orchestration** | [LlamaIndex ≥ 0.14](https://www.llamaindex.ai/) |
| **LLM** | `meta-llama/llama-3.1-70b-instruct` via [OpenRouter API](https://openrouter.ai/) |
| **Embeddings** | `BAAI/bge-base-en-v1.5` (Local CPU) via LlamaIndex HuggingFace Embedding |
| **Vector DB** | [Chroma DB](https://docs.trychroma.com/) (local, persistent storage) |
| **PDF Parser** | `pypdf` |
| **DOCX Parser** | `python-docx` |
| **Fuzzy Matching** | `rapidfuzz` |

---

## Project Structure

```
Resume-RAG-LLM/
├── app.py                   # Streamlit dashboard & styling
├── requirements.txt         # Python dependencies
├── .env.template            # OpenRouter API key template
├── implementation_plan.md   # Design document
├── flowchart_demo.html      # Interactive local demo visualization
│
├── Resumes/                 # Directory to place candidate resumes (PDF/DOCX)
│   ├── ASHOK_Reddy_RESUME - M Ashok reddy.pdf
│   └── ...
│
├── src/
│   ├── config.py            # Configuration loader (.env, model settings)
│   ├── parser.py            # PDF/DOCX parser and filename candidate name extractor
│   ├── indexer.py           # Builds Chroma DB index with local BGE embeddings
│   ├── matcher.py           # Fuzzy candidate name resolver using rapidfuzz
│   ├── classifier.py        # Local query intent classifier (detects comparison queries)
│   ├── retriever.py         # Main FilteredQueryEngine (the 10-step RAG pipeline)
│   └── evaluator.py         # LLM-as-a-judge scorers for 5 RAG metrics
│
└── tests/
    ├── test_retrieval.py    # Retrieval-focused integration tests
    └── test_rag.py          # End-to-end RAG validation tests
```

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- An **OpenRouter API Key** — obtain one at [https://openrouter.ai/](https://openrouter.ai/)

### 1 — Clone the repository

```bash
git clone https://github.com/mmkyadav/Resume-RAG-LLM.git
cd Resume-RAG-LLM
```

### 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### 3 — Configure environment variables

Create a `.env` file from the template:

```bash
copy .env.template .env
```

And set your API key inside `.env`:

```dotenv
# .env
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Optional model overrides:
# LLM_MODEL=meta-llama/llama-3.1-70b-instruct
# EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
```

### 4 — Add resumes

Place your PDF or DOCX resume files in the `Resumes/` directory.

**Required filename format:**
```
<Description> - <Candidate Full Name>.<ext>
```

Examples:
- `ASHOK_Reddy_RESUME - M Ashok reddy.pdf`
- `PAVAN_KAMMA_CV - pavanteja kamma.pdf`
- `Yasawi_resume - yasaswi kotha.docx`

### 5 — Build the vector index

Run the indexer once to process resumes and populate Chroma DB:

```bash
python src/indexer.py
# or force a full rebuild:
python src/indexer.py --rebuild
```

This parses the documents, generates embeddings locally using `bge-base-en-v1.5`, and stores them in `chroma_db/`.

### 6 — Launch the Streamlit app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Usage & Queries

### Asking about a candidate

Type a query mentioning a candidate by name:
- *What are Ashok's Python skills?*
- *Tell me about pavanteja's education.*
- *Summarize yasaswi's projects.*
- *What is Trinadh's contact email?*

### Spelling Corrections

The system handles common spelling variations and shortforms automatically:

| Input | Resolved to |
|---|---|
| `pawan` | `pavanteja kamma` |
| `yasasvi` | `yasaswi kotha` |
| `trinad` | `Trinadh Kumar Reddi` |
| `krish` | `Mungara Muddu Krishna yadav` |
| `shirly` | `21H51A6740-KANDRU SHIRLEY KATHERINE B.Tech-CSD(2021-25)` |

### Blocked Queries

Queries comparing candidates, aggregating across candidates, or not referencing any candidate are rejected to ensure scope safety:
- **Comparison:** `"Compare Ashok and Pawan"`
- **Ranking:** `"Who is the best candidate for Java?"`
- **Listing:** `"List all candidates with React experience"`
- **General/No Name:** `"What is Python?"`

---

## Running Tests

### End-to-end RAG validation tests
```bash
pytest tests/test_rag.py -v
```

### Retrieval pipeline tests
```bash
pytest tests/test_retrieval.py -v
```

---

## Troubleshooting

- **Chroma DB not found/empty:** Run `python src/indexer.py` first to index the resumes.
- **Mock Mode response:** If the OpenRouter API key is missing or is the default placeholder, the system defaults to mock mode. Add your valid key to `.env`.
- **First query slow:** On first run, the local embedding model `bge-base-en-v1.5` (~440MB) is downloaded. Subsequent runs use the local cache and are fast.
