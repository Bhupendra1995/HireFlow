# HireFlow — Intelligent Candidate Search and Evaluation

HireFlow is an AI-powered recruitment system that combines **semantic search** with **Generative AI** to intelligently match candidates to job descriptions. It goes beyond keyword filtering by understanding the meaning and context of both resumes and job postings, ranking candidates by confidence score and providing human-readable insights into their strengths, gaps, and overall fit.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        INDEXING  (one-time)                             │
│                                                                         │
│   PDF Resumes                                                           │
│       │                                                                 │
│       ▼                                                                 │
│   [PyMuPDF Parser]      extract raw text from each PDF                  │
│       │                                                                 │
│       ▼                                                                 │
│   [LlamaIndex SentenceSplitter]   chunk_size=256, overlap=32            │
│       │                                                                 │
│       ▼                                                                 │
│   [Gemini embedding-001]   3072-dim vectors per chunk                   │
│       │                                                                 │
│       ▼                                                                 │
│   [Pinecone Serverless Index]   vectors stored with metadata            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │  index ready
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        RETRIEVAL PIPELINE                               │
│                                                                         │
│   Job Description  ──►  [Query Expansion]  →  sub-queries per section  │
│         │                      │                                        │
│         │                      ▼                                        │
│         │           [Gemini embedding-001]  →  dense query vector       │
│         │                      │                                        │
│         │         ┌────────────┴────────────┐                          │
│         │         ▼                         ▼                          │
│         │   [Pinecone Dense Search]   [BM25 Sparse Search]             │
│         │      top-K×2 chunks          top-K×2 chunks                  │
│         │         │                         │                          │
│         │         └────────────┬────────────┘                          │
│         │                      ▼                                        │
│         │         [Reciprocal Rank Fusion]   merge dense + sparse       │
│         │                      │                                        │
│         │                      ▼                                        │
│         └──────►  [Cross-Encoder Reranker]   score chunks vs JD        │
│                               │                                         │
│                               ▼                                         │
│                  [MMR Deduplication]   max 2 chunks per candidate       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     SELF-RAG LOOP  (max 3 retries)                      │
│                                                                         │
│   [Gemini 2.5-flash]  ◄─────────────────────────────────────────────┐  │
│         │  generate evaluation                                        │  │
│         ▼                                                             │  │
│   [Self-RAG Critique]                                                 │  │
│         │                                                             │  │
│         ├── IsRelevant?  (chunk useful?)    NO  → discard chunk       │  │
│         │                                                             │  │
│         ├── IsSupported? (grounded?)        NO  → lower temp, retry ─┘  │
│         │                                                                │
│         └── IsUseful?   (score 1–5)         < 3 → LOW confidence flag   │
│                                                                          │
│                               │  pass                                   │
│                               ▼                                          │
│   [PydanticOutputParser]   score · strengths · gaps · summary           │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                  RAGAS ADAPTIVE LOOP  (max 5 iterations)                │
│                                                                         │
│   [RAGAS Evaluation]                                                    │
│         │                                                               │
│         ├── Context Precision  < 0.70  →  Reduce K + Add Reranker       │
│         │                                                               │
│         ├── Context Recall     < 0.75  →  Increase K + Hybrid Search    │
│         │                                  + Query Expansion            │
│         │                                                               │
│         ├── Faithfulness       < 0.80  →  Harden Prompt + Lower Temp    │
│         │                                                               │
│         └── Answer Relevancy   < 0.75  →  MMR Dedup + Restructure       │
│                                            Prompt                       │
│                               │                                         │
│              ┌────────────────┴─────────────────────┐                   │
│              ▼                                       ▼                   │
│     All metrics pass                    Adjust PipelineConfig            │
│              │                          re-run retrieval + eval ─────►  │
│              ▼                                                           │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │       RANKED RESULTS           │
                    │   score · strengths · gaps     │
                    │       summary · flags          │
                    └───────────┬───────────────────┘
                                │
               ┌────────────────┴────────────────┐
               ▼                                 ▼
     [Streamlit UI]                          [CLI]
     3-page recruiter dashboard        python main.py search
```

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **LLM** | `gemini-2.5-flash` (free tier) | Candidate evaluation + Self-RAG critique |
| **Embeddings** | `gemini-embedding-001` (free tier) | Resume + JD vector generation (3072-dim) |
| **Vector DB** | Pinecone Serverless (free Starter) | Resume vector storage and similarity search |
| **RAG Framework** | LlamaIndex | Document parsing, chunking, retrieval |
| **Orchestration** | LangChain | Prompt templates, chains, output parsing |
| **Sparse Retrieval** | rank-bm25 | Local BM25 for hybrid search (no API cost) |
| **Reranker** | sentence-transformers `ms-marco-MiniLM-L-6-v2` | Local cross-encoder reranking (no API cost) |
| **PDF Parsing** | PyMuPDF (fitz) | Extract text from resume PDFs |
| **Evaluation** | RAGAS | Context Precision, Recall, Faithfulness, Answer Relevancy |
| **UI** | Streamlit | 3-page recruiter dashboard |
| **Validation** | Pydantic v2 | Typed data models throughout the pipeline |
| **Config** | python-dotenv | API keys and environment settings |

---

## Project Structure

```
HireFlow/
├── data/
│   ├── resumes/                    ← PDF resume corpus (50 test resumes)
│   └── job_descriptions/           ← JD text files (5 test JDs)
│
├── scripts/
│   └── generate_test_data.py       ← Generates 50 resumes + 5 JDs as PDFs/TXT
│
├── src/
│   ├── models/
│   │   └── schemas.py              ← Pydantic models: ResumeChunk, CandidateEvaluation,
│   │                                  SelfRAGResult, RAGASScores, PipelineConfig, etc.
│   ├── ingestion/
│   │   └── pdf_parser.py           ← PDF parsing, chunking, domain inference, JD loading
│   ├── embeddings/
│   │   └── embedder.py             ← Gemini embedding + Pinecone index management
│   ├── retrieval/
│   │   └── retriever.py            ← Hybrid search, reranker, MMR dedup, query expansion
│   └── evaluation/
│       └── evaluator.py            ← LangChain chains, Self-RAG loop, RAGAS adaptive loop
│
├── app/
│   └── streamlit_app.py            ← 3-page Streamlit recruiter UI
│
├── main.py                         ← CLI interface (same pipeline as UI)
├── requirements.txt
├── .env.example                    ← Copy to .env and fill in API keys
└── .gitignore
```

---

## Setup

### Prerequisites
- Python 3.12+
- Gemini API key (free — [get one at Google AI Studio](https://aistudio.google.com/apikey))
- Pinecone API key (free Starter plan — [pinecone.io](https://www.pinecone.io))

### 1. Clone and create virtual environment

```powershell
git clone <repo-url>
cd HireFlow
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Configure environment

```powershell
Copy-Item .env.example .env
```

Edit `.env` and fill in your keys:

```env
GEMINI_API_KEY=your_gemini_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
```

### 4. Generate test data (50 resumes + 5 JDs)

```powershell
python scripts/generate_test_data.py
```

---

## CLI Usage

### Check status

```powershell
python main.py status
```
```
======================================================================
  HIREFLOW STATUS
======================================================================
  Resumes in corpus : 50
  JDs available     : 5
  Pinecone indexed  : NO — run: python main.py index
======================================================================
```

### Index resumes into Pinecone (run once)

```powershell
python main.py index
```

Force re-index (wipe and rebuild):

```powershell
python main.py index --force
```

### List available JD files

```powershell
python main.py list-jds
```

### Search with a JD file

```powershell
python main.py search --jd-file jd_ml_engineer.txt
```

### Search with inline JD text

```powershell
python main.py search --jd-text "Looking for a Senior Python Backend Developer with FastAPI and PostgreSQL experience..."
```

### Interactive search (paste JD at prompt)

```powershell
python main.py search
```

### Search with RAGAS evaluation

```powershell
python main.py search --jd-file jd_ml_engineer.txt --ragas
```

### Full options

```powershell
python main.py search --jd-file jd_devops_engineer.txt `
    --top-k 12 `
    --top-n 5 `
    --temperature 0.2 `
    --ragas
```

| Flag | Default | Description |
|---|---|---|
| `--jd-file` | — | JD filename from `data/job_descriptions/` or absolute path |
| `--jd-text` | — | JD passed directly as a string |
| `--top-k` | 8 | Number of chunks to retrieve from Pinecone |
| `--top-n` | 10 | Number of candidates to display in output |
| `--temperature` | 0.3 | Gemini LLM temperature (lower = more grounded) |
| `--no-hybrid` | off | Disable BM25 hybrid search |
| `--no-reranker` | off | Disable cross-encoder reranker |
| `--no-compression` | off | Disable context compression |
| `--no-mmr` | off | Disable MMR deduplication |
| `--ragas` | off | Run RAGAS evaluation after search |

---

## Streamlit UI

```powershell
streamlit run app/streamlit_app.py
```

The UI has 3 pages:

| Page | Purpose |
|---|---|
| **Resume Management** | Upload PDFs, trigger indexing, re-index corpus |
| **Candidate Search** | Paste or load a JD, configure pipeline, view ranked candidates with strengths/gaps/summary, conversational follow-up Q&A |
| **Evaluation Dashboard** | View RAGAS metric scores with threshold bars, see which metrics failed and what config was auto-applied |

---

## How the Pipeline Works

### Retrieval (Hybrid Search)
1. JD text is embedded using `gemini-embedding-001` → dense query vector
2. Pinecone returns top-K×2 resume chunks by cosine similarity (dense)
3. BM25 returns top-K×2 resume chunks by keyword overlap (sparse)
4. Reciprocal Rank Fusion merges both rankings
5. Cross-encoder reranker scores each chunk against the full JD text
6. MMR deduplication caps each candidate at 2 chunks for diversity

### Self-RAG Loop (max 3 retries)
Each candidate goes through:
1. **Generate** — Gemini evaluates candidate against JD → structured output
2. **Critique** — Gemini checks: `IsRelevant`, `IsSupported`, `UsefulnessScore`
3. If `IsSupported=false` → lower temperature, retry
4. After 3 failed attempts → return best result with `LOW` confidence flag

### RAGAS Adaptive Loop (max 5 iterations)
After evaluation, RAGAS scores 4 metrics against thresholds:

| Metric | Threshold | If Failing → Auto-Fix |
|---|---|---|
| Context Precision | ≥ 0.70 | Reduce top-K, enable reranker + compression |
| Context Recall | ≥ 0.75 | Increase top-K, enable hybrid search + query expansion |
| Faithfulness | ≥ 0.80 | Reduce LLM temperature, enable compression |
| Answer Relevancy | ≥ 0.75 | Enable MMR deduplication |

Config is adjusted and pipeline re-runs until all metrics pass or 5 iterations are reached.

---

## Rate Limits (Free Tier)

| Service | Limit | How HireFlow Handles It |
|---|---|---|
| Gemini LLM | 15 RPM | Candidate evaluations batched in groups of 5 with 4.5s delay |
| Gemini Embeddings | 100 RPM | Chunks embedded in batches of 10 with 1s delay |
| Pinecone Starter | 2GB storage | ~10,000 resume chunks fit comfortably |
