# 🧠 AskMyNotes

<div align="center">

**Upload notes. Ask questions. Get AI answers.**

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Endee](https://img.shields.io/badge/Endee-Vector_DB-00C4A7?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Gemini](https://img.shields.io/badge/Gemini-AI-4285F4?style=for-the-badge&logo=google&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

AskMyNotes is a production-style AI knowledge assistant. Upload your PDF notes, then ask any question — **Gemini, Claude, GPT-4o, and Ollama all answer simultaneously**, side-by-side, grounded in your documents via the **Endee vector database**.

[What It Does](#-what-this-project-does) · [Architecture](#-system-architecture) · [Quick Start](#-quick-start) · [How Endee Is Used](#-how-endee-is-used)

</div>

---

## 📌 What This Project Does

AskMyNotes turns your PDF notes into a searchable AI knowledge base. The moment you upload a PDF:

1. Text is extracted page-by-page using `pdfplumber`
2. Pages are split into overlapping chunks to preserve context
3. Each chunk is converted into a 384-dimensional semantic vector using `all-MiniLM-L6-v2`
4. All vectors are stored and indexed in **Endee** running locally via Docker
5. When you ask a question, Gemini rewrites it for better retrieval, then Endee finds the most relevant passages using HNSW cosine similarity search
6. A cross-encoder reranks the results for accuracy
7. The top passages are injected into prompts sent **simultaneously** to every configured LLM
8. All answers appear side-by-side in a yupp.ai-style card layout

---

## 🏗 System Architecture

```
╔══════════════════════════════════════════════════════════════════╗
║  INDEXING  (runs on every PDF upload)                            ║
║                                                                  ║
║  PDF Upload → pdfplumber (page-by-page) → chunk_text()          ║
║       → all-MiniLM-L6-v2 (384-dim) → Endee vector index         ║
╚══════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════╗
║  QUERY PIPELINE  (runs on every question)                        ║
║                                                                  ║
║  User question                                                   ║
║      │                                                           ║
║      ▼                                                           ║
║  query_rewriter.py ── Gemini rewrites for better retrieval       ║
║      │  (standalone · expand · hyde strategies)                  ║
║      ▼                                                           ║
║  Endee HNSW search ── fetches top-15 candidates (cosine)         ║
║      │                                                           ║
║      ▼                                                           ║
║  reranker.py ── cross-encoder keeps best 5 passages              ║
║      │                                                           ║
║      ▼                                                           ║
║  chat_memory.py ── injects conversation history into prompt      ║
║      │                                                           ║
║      ▼                                                           ║
║  llm_clients.py ── fires ALL LLMs simultaneously (threads)       ║
║      │                                                           ║
║      ▼                                                           ║
║  app.py ── side-by-side answer cards  (AskMyNotes UI)         ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## 📁 Project Structure

```
askmynotes/
│
├── data/
│   └── documents.txt              ← Optional static knowledge base
│
├── ── INGESTION ───────────────────────────────────────────────────
│
├── embed_documents.py             ← Index documents.txt into Endee
├── pdf_loader.py                  ← PDF extraction + ingestion into Endee
│
├── ── RETRIEVAL ───────────────────────────────────────────────────
│
├── search_documents.py            ← SemanticSearch class (wraps Endee)
├── query_rewriter.py              ← Gemini query rewriting (3 strategies)
├── reranker.py                    ← Cross-encoder reranking
│
├── ── GENERATION ──────────────────────────────────────────────────
│
├── llm_clients.py                 ← All LLM backends + parallel execution
├── chat_memory.py                 ← Sliding-window conversation memory
├── enhanced_rag_pipeline.py       ← Orchestrates the full pipeline
│
├── ── UI ──────────────────────────────────────────────────────────
│
├── app.py                      ← AskMyNotes — main Streamlit app
│
├── ── CONFIG ──────────────────────────────────────────────────────
│
├── config.py                      ← All settings (Endee, LLMs, retrieval)
├── requirements.txt            ← All dependencies
└── .env.example                   ← Environment variable template
```

---

## ✨ Features

### 📄 PDF Knowledge Base
- Upload any number of PDFs directly from the browser
- Text extracted page-by-page with `pdfplumber`
- Chunks embedded and stored in Endee with full metadata: filename, page number, chunk index
- Multiple PDFs live in the same Endee index — all searchable together
- Document manager in sidebar shows all uploaded PDFs with chunk counts

### 🔵 Endee Vector Retrieval
- HNSW cosine similarity search — sub-5ms query latency
- Fetches 15 candidates per query (configurable via `RERANK_FETCH_K`)
- Runs entirely locally via Docker — no cloud dependency

### ✏️ Gemini Query Rewriting
Three strategies that improve retrieval quality before querying Endee:

| Strategy | What it does | Best for |
|----------|-------------|----------|
| `standalone` | Uses chat history to make follow-ups self-contained | "tell me more" → "tell me more about HNSW indexing" |
| `expand` | Adds synonyms and related keywords | Short queries like "RAG" or "embeddings" |
| `hyde` | Generates a hypothetical answer, uses it as the query vector | Factual questions — HyDE embeds closer to real passages |
| `combined` | Standalone then expand | Best overall recall |

### 📊 Cross-Encoder Reranking
- Endee fetches 15 candidates using the fast bi-encoder
- `cross-encoder/ms-marco-MiniLM-L-6-v2` re-scores all 15 by reading query + passage together
- Top 5 by rerank score are sent to the LLMs
- ~20–30% improvement in answer relevance with only ~130ms added latency

### ⚡ Parallel Multi-LLM (yupp.ai style)
- Every question fires **all configured LLMs simultaneously** using `ThreadPoolExecutor`
- Total time ≈ slowest single LLM, not the sum of all
- Answers appear as equal-width cards in a single row
- Each card shows LLM name, response time, and the full answer
- Failed/offline LLMs show a greyed error card without blocking the others

### 🧠 Chat Memory
- Sliding window of last 6 conversation turns (configurable via `MEMORY_MAX_TURNS`)
- History injected into both the query rewriter and LLM prompts
- Follow-up questions work naturally: "what about its cost?" resolves correctly
- Clear memory with one sidebar button

### 📚 Source Citations
- Every answer shows which documents and pages were retrieved
- Cosine similarity score and cross-encoder rerank score shown per source
- Visual score bar for quick relevance scanning
- Full passage text inside an expandable source panel

### 📐 Retrieval Metrics
Each question displays timing chips:
`✏️ rewrite` · `🔍 retrieve` · `📊 rerank` · `⏱ total`

---

## ⚡ Quick Start

### Prerequisites
- Python 3.8+
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)

### 1 — Clone

```bash
git clone https://github.com/YOUR_USERNAME/askmynotes.git
cd askmynotes
```

### 2 — Virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### 4 — Start Endee with Docker

Create `docker-compose.yml`:

```yaml
services:
  endee:
    image: endeeio/endee-server:latest
    container_name: endee-server
    ports:
      - "8080:8080"
    volumes:
      - endee-data:/data
    restart: unless-stopped
volumes:
  endee-data:
```

```bash
docker compose up -d
```

Verify Endee is running at **http://localhost:8080**

### 5 — Configure environment

```bash
cp .env.example .env
```

Minimum `.env` to get started:

```env
GEMINI_API_KEY=your_gemini_key_here
MULTI_LLM_PROVIDERS=gemini,anthropic
ANTHROPIC_API_KEY=your_anthropic_key_here
```

> 💡 Get a **free** Gemini key at https://makersuite.google.com/app/apikey
> 💡 No API keys? Set `MULTI_LLM_PROVIDERS=mock` to test without any.

### 6 — Launch AskMyNotes

```bash
streamlit run app.py
```

Open **http://localhost:8501** — the upload screen appears. Drop in a PDF, click **"Start asking questions"**, then ask anything.

---

## 🖥 UI Flow

```
① Upload screen  (shown on first visit — chat is hidden)
   ┌─────────────────────────────────────┐
   │  🧠 AskMyNotes                      │
   │  Upload notes. Ask questions.       │
   │  Get AI answers.                    │
   │                                     │
   │  ┌── Drop PDF files here ─────────┐ │
   │  │  📄 ML_notes.pdf   ✅ indexed  │ │
   │  │  📄 RAG_guide.pdf  ✅ indexed  │ │
   │  └────────────────────────────────┘ │
   │                                     │
   │    ⚡ Start asking questions →      │
   └─────────────────────────────────────┘
                  ↓ click button

② Chat arena  (all LLMs fire on every question)
   ┌──────────────────────────────────────────────────────────────┐
   │  Q: What is HNSW indexing?                                   │
   │  ✏️ Rewritten: What is HNSW and how does it work in Endee?   │
   │                                                              │
   │  ┌─ Gemini ──┐  ┌─ Claude ──┐  ┌─ GPT-4o ─┐  ┌─ Ollama ─┐ │
   │  │ HNSW is… │  │ HNSW is… │  │ HNSW is… │  │ HNSW is… │ │
   │  │  1.2s    │  │  1.5s    │  │  1.8s    │  │  3.1s    │ │
   │  └──────────┘  └──────────┘  └──────────┘  └──────────┘ │
   │                                                              │
   │  ✏️ 340ms  🔍 38ms  📊 122ms  ⏱ 3.2s total                 │
   │  📚 5 sources ▼                                             │
   └──────────────────────────────────────────────────────────────┘
```

---

## 🤖 LLM Providers

| Provider | Value in `.env` | API Key |
|----------|----------------|---------|
| Google Gemini | `gemini` | `GEMINI_API_KEY` — [free key](https://makersuite.google.com/app/apikey) |
| Anthropic Claude | `anthropic` | `ANTHROPIC_API_KEY` |
| OpenAI GPT-4o | `openai` | `OPENAI_API_KEY` |
| Ollama (local/free) | `ollama` | None — [install Ollama](https://ollama.com) |
| Mock (testing) | `mock` | None |

Configure which LLMs fire via `MULTI_LLM_PROVIDERS` in `.env`:

```env
# Run all four
MULTI_LLM_PROVIDERS=gemini,anthropic,openai,ollama

# Gemini + Claude only
MULTI_LLM_PROVIDERS=gemini,anthropic

# Fully offline, no API keys
MULTI_LLM_PROVIDERS=ollama
OLLAMA_MODEL=llama3
```

Toggle individual LLMs on/off in the sidebar checkboxes at any time.

---

## ⚙️ Configuration Reference

All settings in `config.py`, overridable via `.env`:

| Setting | Default | Description |
|---------|---------|-------------|
| `ENDEE_BASE_URL` | `http://localhost:8080/api/v1` | Endee server address |
| `ENDEE_INDEX_NAME` | `rag_documents` | Vector index name |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | HuggingFace embedding model |
| `EMBEDDING_DIMENSION` | `384` | Vector dimensions |
| `CHUNK_SIZE` | `400` | Characters per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `TOP_K_RESULTS` | `5` | Passages kept after reranking |
| `RERANK_FETCH_K` | `15` | Candidates fetched before reranking |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder model |
| `GEMINI_MODEL` | `gemini-1.5-flash` | Gemini model for answers |
| `GEMINI_REWRITE_MODEL` | `gemini-1.5-flash` | Gemini model for query rewriting |
| `MULTI_LLM_PROVIDERS` | `gemini,anthropic` | LLMs to fire in parallel |
| `MULTI_LLM_TIMEOUT` | `60` | Seconds before an LLM times out |
| `MEMORY_MAX_TURNS` | `6` | Conversation turns kept in memory |

---

## 🧩 How Each Module Works

### `embed_documents.py`
Indexes the optional `data/documents.txt` static knowledge base into Endee. Handles connection, index creation (or reuse), chunking, embedding, and upsert. Run once to seed the knowledge base.

### `pdf_loader.py`
Called automatically when a PDF is uploaded in the UI. Extracts text page-by-page with `pdfplumber`, runs the identical chunking and embedding logic as `embed_documents.py`, and upserts vectors into Endee with `{filename, page, source="pdf"}` metadata. The retrieval pipeline needs no changes — it just sees more vectors.

### `search_documents.py`
Wraps the Endee Python SDK into a clean `SemanticSearch` class. Lazy-loads the embedding model and Endee index on first call. Exposes a single `search(query, top_k)` method that returns normalised result dicts with text, title, similarity score, filename, and page.

### `query_rewriter.py`
Uses Gemini to rewrite the user's question before it reaches Endee. Three strategies — `standalone` (resolves follow-ups using chat history), `expand` (adds related keywords), `hyde` (generates a hypothetical answer whose vector is closer to real document passages). Gracefully returns the original query if Gemini is unavailable.

### `reranker.py`
Wraps `sentence_transformers.CrossEncoder`. Receives the query and all retrieved chunks, scores each `(query, chunk)` pair jointly (not independently like the bi-encoder), and returns the top-K by score. Adds a `rerank_score` field to each chunk — the rest of the pipeline is unchanged.

### `llm_clients.py`
Houses all LLM backends — Gemini, Claude, GPT-4o, Ollama, Mock — under a single `call_llm(prompt, provider)` router. Provides `call_llm_parallel(prompt, providers)` which submits all providers to a `ThreadPoolExecutor` simultaneously. Total wall time equals the slowest single LLM, not their sum.

### `chat_memory.py`
Stores the last N conversation turns in memory. Provides `as_list()` for the query rewriter and `build_prompt_with_memory()` which injects prior exchanges into the LLM prompt so follow-up questions resolve correctly without repeating context.

### `enhanced_rag_pipeline.py`
The main orchestrator. `run_parallel(query, providers)` executes the full pipeline: rewrite → Endee search → rerank → build context + memory prompt → fire all LLMs in parallel → update memory. Returns a single result dict with all answers, sources, and timing metrics.

### `app.py` — AskMyNotes
The Streamlit UI. **Screen 1** (upload gate): centered PDF uploader, per-file progress bars, "Start asking" button — the chat input is completely hidden until at least one PDF is indexed. **Screen 2** (chat arena): skeleton loading cards appear immediately while LLMs are thinking, then replaced by real answer cards side-by-side. Sidebar has LLM toggles, pipeline settings, document manager, and session stats.

---

## 🔵 How Endee Is Used

Endee is the retrieval engine that makes every answer factual. Three SDK calls:

**1. Create the index** (once, on startup):
```python
from endee import Endee, Precision

client = Endee()
client.set_base_url("http://localhost:8080/api/v1")
client.create_index(
    name="rag_documents",
    dimension=384,
    space_type="cosine",
    precision=Precision.INT8,
)
```

**2. Upsert vectors** (on every PDF upload):
```python
index = client.get_index(name="rag_documents")
index.upsert([{
    "id":     "pdf_a1b2_p0001_c000",
    "vector": [0.12, -0.34, ...],       # 384 floats from all-MiniLM-L6-v2
    "meta":   {
        "text":     "HNSW builds a multi-layer graph...",
        "title":    "ML_notes.pdf — Page 4",
        "filename": "ML_notes.pdf",
        "page":     4,
        "source":   "pdf",
    },
}])
```

**3. Query at search time** (on every question):
```python
results = index.query(vector=query_embedding, top_k=15, ef=128)
# Returns the 15 most semantically similar passages in < 5ms
```

---

## 🐛 Troubleshooting

| Problem | Fix |
|---------|-----|
| Cannot connect to Endee | `docker compose up -d` · verify at http://localhost:8080 |
| PDF shows no extracted text | PDF may be scanned/image-only — use a text-selectable PDF |
| Gemini API error | Check `GEMINI_API_KEY` in `.env` · get free key at makersuite.google.com |
| LLM card shows error | That provider is offline or misconfigured — other cards still work |
| Reranker slow on first query | Cross-encoder model downloads once on first use (~50 MB) |
| `pdfplumber` not found | Run `pip install -r requirements.txt` |

---

## 🚀 Potential Extensions

- [ ] OCR support for scanned/image PDFs (`pytesseract`)
- [ ] Streaming LLM responses — token-by-token inside each card
- [ ] Per-LLM answer quality voting (thumbs up/down)
- [ ] Export conversation as a PDF or Markdown report
- [ ] User authentication with per-user knowledge bases
- [ ] Cloud deployment of Endee (AWS / GCP / Azure)

---

## 📚 References

- [Endee GitHub](https://github.com/endee-io/endee)
- [Endee Documentation](https://docs.endee.io)
- [sentence-transformers](https://www.sbert.net)
- [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [Google Gemini API](https://ai.google.dev)
- [Anthropic Claude](https://console.anthropic.com)
- [pdfplumber](https://github.com/jsvine/pdfplumber)

---

## 📄 License

MIT — free to use, modify, and distribute.

---

<div align="center">

**🧠 AskMyNotes — Upload notes. Ask questions. Get AI answers.**

Built with [Endee Vector Database](https://github.com/endee-io/endee)

</div>
