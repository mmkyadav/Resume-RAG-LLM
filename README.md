# Resume RAG LLM 🤖

A **Retrieval-Augmented Generation (RAG)** system that answers detailed questions about individual candidates from a directory of resumes. Built with **LlamaIndex**, **Chroma DB**, **Google Gemini LLMs**, and a premium **Streamlit** interface.

---

## Features

| Feature | Description |
|---|---|
| 🔍 **Single-Candidate QA** | Deep-dive into any candidate's skills, experience, education, or projects |
| ✏️ **Spelling Correction** | "pawan" → "pavanteja kamma", "yasasvi" → "yasaswi kotha" |
| 🚫 **Multi-Resume Blocking** | Comparison, ranking, and aggregation queries are automatically rejected |
| 💬 **Chat History** | Full conversation history with per-message metadata |
| 📊 **Live Stats** | Query counts, correction alerts, and blocked-query counters |
| 🎨 **Premium Dark UI** | Glassmorphism design with smooth animations |
| 🔑 **In-App API Key Input** | Enter your Gemini API key directly in the sidebar — no `.env` edit needed |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    app.py  (Streamlit UI)                   │
│   Chat window · Stats · Spelling alerts · Blocked alerts    │
│   🔑 Gemini API Key input in sidebar (live engine re-init)  │
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
   Gemini LLM or          rapidfuzz             Build Chroma index
   heuristic fallback     fuzzy matching         with Gemini Embeddings
```

---

## Tech Stack

| Component | Technology |
|---|---|
| **Frontend** | [Streamlit](https://streamlit.io/) |
| **Orchestration** | [LlamaIndex ≥ 0.10](https://www.llamaindex.ai/) |
| **LLM** | `models/gemini-1.5-flash` via [Google Gemini API](https://ai.google.dev/) |
| **Embeddings** | `models/text-embedding-004` via Google Gemini API |
| **Vector DB** | [Chroma DB](https://docs.trychroma.com/) (local, file-based) |
| **PDF Parser** | `pypdf` |
| **DOCX Parser** | `python-docx` |
| **Fuzzy Matching** | `rapidfuzz` |

---

## Project Structure

```
resume-rag-team3/
├── app.py                   # Streamlit dashboard (Person 3)
├── requirements.txt         # Python dependencies
├── .env.template            # API key template
├── implementation_plan.md   # Team design document
│
├── Resumes/                 # Drop PDF/DOCX resumes here
│   ├── ASHOK_Reddy_RESUME - M Ashok reddy.pdf
│   └── ...
│
├── src/
│   ├── config.py            # Config loader (.env + paths, Gemini defaults)
│   ├── parser.py            # PDF/DOCX parser + name extractor
│   ├── indexer.py           # Build/refresh Chroma DB index (Gemini Embeddings)
│   ├── matcher.py           # Fuzzy name matching
│   ├── classifier.py        # Gemini LLM query classifier (+ fallback)
│   └── retriever.py         # Filtered LlamaIndex query engine
│
└── tests/
    ├── test_retrieval.py    # Person 2 retrieval tests
    └── test_rag.py          # Person 3 — end-to-end RAG tests
```

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- A **Google Gemini API key** — get one free at [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)

### 1 — Clone the repository

```bash
git clone https://github.com/mmkyadav/Resume-RAG-LLM.git
cd Resume-RAG-LLM
```

### 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### 3 — Configure environment (optional)

You can set your API key in `.env` **or** paste it directly in the sidebar of the Streamlit app at runtime.

```dotenv
# .env
GEMINI_API_KEY=AIzaSy_your_key_here

# Optional model overrides:
# LLM_MODEL=models/gemini-1.5-pro
# EMBEDDING_MODEL=models/text-embedding-004
```

> **Tip:** If you leave `.env` empty, you can still use the app by pasting your key in the **🔑 Gemini API Key** field in the sidebar.

### 4 — Add resumes

Place your PDF or DOCX resume files in the `Resumes/` directory.

**Required filename format:**
```
<Description> - <Candidate Full Name>.<ext>
```

Examples:
```
ASHOK_Reddy_RESUME - M Ashok reddy.pdf
PAVAN_KAMMA_CV - pavanteja kamma.pdf
Yasaswi_Profile - yasaswi kotha.docx
```

### 5 — Build the vector index

Run the indexer **once** (or whenever resumes change). This requires your Gemini API key to be set (via `.env` or `--api-key` flag):

```bash
python src/indexer.py
# or force a full rebuild:
python src/indexer.py --rebuild
```

This parses all resumes, generates embeddings via the Gemini Embedding API, and stores them in `chroma_db/`.

### 6 — Launch the Streamlit app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

Paste your **Gemini API key** in the **🔑 Gemini API Key** field in the left sidebar to enable live LLM responses.

---

## Usage

### Asking about a candidate

Type a question mentioning a candidate by name:

```
What are Ashok's Python skills?
Tell me about pavanteja's education
Summarize yasasvi's projects
What is Trinadh's contact email?
```

### Spelling corrections

The system automatically corrects common spelling variants and shortforms:

| Input | Resolved to |
|---|---|
| `pawan` | `pavanteja kamma` |
| `yasasvi` | `yasaswi kotha` |
| `trinad` | `Trinadh Kumar Reddi` |
| `krish` | `Mungara Muddu Krishna yadav` |
| `shirly` | `KANDRU SHIRLEY KATHERINE` |

A yellow alert banner will inform you of any applied correction.

### Blocked queries (by design)

The following query types are **rejected** with an explanatory message:

- **Comparison:** `"Compare Ashok and Pawan"`
- **Ranking:** `"Who is the best candidate for Java?"`
- **Listing/Aggregation:** `"List all candidates with React experience"`
- **Shortlisting:** `"Shortlist top 3 for data science"`
- **No name mentioned:** `"What is Python?"`

---

## Running Tests

### End-to-end RAG tests (Person 3)

```bash
pytest tests/test_rag.py -v
```

### Retrieval pipeline tests (Person 2)

```bash
pytest tests/test_retrieval.py -v
```

### All tests

```bash
pytest tests/ -v
```

> **Note:** Tests that require the `Resumes/` directory to be populated will be automatically **skipped** if no resume files are present, and will run in full when the directory contains resume files.

---

## Team & Branch Strategy

| Person | Branch | Responsibility |
|---|---|---|
| **Person 1** | `dev-person-1-ingestion` | Config, parsers, Chroma DB indexer |
| **Person 2** | `dev-person-2-retrieval` | Fuzzy matcher, classifier, filtered retriever |
| **Person 3** | `dev-person-3-frontend` | Streamlit UI, chat integration, tests, README |

All branches merge into `main` sequentially.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `GEMINI_API_KEY is not set` | Add key to `.env` or paste it in the sidebar 🔑 |
| `Resumes directory does not exist` | Create `Resumes/` and add files |
| `Chroma DB not found / empty` | Run `python src/indexer.py` first |
| `Mock Mode` responses | API key missing/invalid — add valid key to `.env` or sidebar |
| Import errors | Run `pip install -r requirements.txt` |

---

## License

MIT License. See [LICENSE](LICENSE) for details.
