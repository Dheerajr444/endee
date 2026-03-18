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

</div>

---

## 📌 What This Project Does

AskMyNotes turns your PDF notes into a searchable AI knowledge base. The moment you upload a PDF:

1. It is extracted page-by-page and split into overlapping text chunks
2. Each chunk is converted into a 384-dimensional semantic vector using `all-MiniLM-L6-v2`
3. All vectors are stored and indexed in **Endee** running locally via Docker
4. When you ask a question, your query is rewritten by Gemini for better retrieval, then Endee finds the most relevant passages using HNSW cosine similarity search
5. The retrieved passages are reranked by a cross-encoder for accuracy, then injected into prompts sent **simultaneously** to every configured LLM
6. All answers appear side-by-side in a yupp.ai-style card layout

---

## 🏗 System Architecture

```
╔══════════════════════════════════════════════════════════════════╗
║  INDEXING  (on every PDF upload)                                 ║
║                                                                  ║
║  PDF Upload → pdfplumber extraction → chunk_text()              ║
║       → all-MiniLM-L6-v2 embeddings → Endee vector index        ║
╚══════════════════════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════════════════════╗
║  QUERY PIPELINE  (on every question)                             ║
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
║  app_v2.py ── side-by-side answer cards  (AskMyNotes UI)         ║
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
├── ── CORE PIPELINE ──────────────────────────────────────────────
│
├── embed_documents.py             ← Index documents.txt into Endee
├── search_documents.py            ← SemanticSearch class (Endee queries)
├── rag_pipeline.py                ← Original single-LLM RAG pipeline
│
├── ── V2 EXTENSIONS ──────────────────────────────────────────────
│
├── pdf_loader.py                  ← PDF extraction + auto-ingestion into Endee
├── query_rewriter.py              ← Gemini query rewriting (3 strategies)
├── reranker.py                    ← Cross-encoder reranking
├── llm_clients.py                 ← All LLM backends + parallel execution
├── chat_memory.py                 ← Sliding-window conversation memory
├── enhanced_rag_pipeline.py       ← Full v2 pipeline (wraps all above)
│
├── ── UI ──────────────────────────────────────────────────────────
│
├── app.py                         ← Original single-LLM Streamlit UI
├── app_v2.py                      ← AskMyNotes — yupp-style parallel UI
├── chatbot.py                     ← Terminal chatbot (/help, /sources, etc.)
│
├── ── CONFIG ──────────────────────────────────────────────────────
│
├── config.py                      ← All settings (Endee, LLMs, retrieval)
├── requirements.txt               ← v1 dependencies
├── requirements_v2.txt            ← v2 dependencies (adds Gemini, reranker)
└── .env.example                   ← Environment variable template
```

---

## ✨ Features

### 🔵 Core RAG Pipeline
- **Endee vector database** — HNSW cosine similarity search, sub-5ms latency, runs locally via Docker
- **all-MiniLM-L6-v2** embeddings — 384-dimensional semantic vectors, fully offline
- **Overlapping text chunking** — preserves context at chunk boundaries
- **Metadata stored per chunk** — source filename, page number, chunk index

### 📄 PDF Knowledge Base
- Upload any PDF directly from the browser
- Text extracted page-by-page with `pdfplumber`
- Each page chunked and embedded using the same pipeline as static documents
- Filename and page number stored in Endee metadata — every answer cites its source
- Multiple PDFs stored in the same index — all searchable together

### ✏️ Gemini Query Rewriting
Three strategies to improve retrieval quality before querying Endee:

| Strategy | What it does | Best for |
|----------|-------------|---------|
| `standalone` | Uses chat history to make follow-up questions self-contained | "tell me more about it" → "tell me more about HNSW indexing" |
| `expand` | Adds synonyms and related keywords | Short queries like "RAG" or "embeddings" |
| `hyde` | Generates a hypothetical answer, uses that as the query (HyDE) | Factual questions — hypothetical answers embed closer to real passages |
| `combined` | Standalone then expand | Best overall recall |

### 📊 Cross-Encoder Reranking
- Endee fetches 15 candidates (bi-encoder, fast)
- `cross-encoder/ms-marco-MiniLM-L-6-v2` re-scores all 15 by reading query + passage together
- Top 5 by rerank score are sent to the LLM
- ~20–30% improvement in answer relevance with only ~130ms added latency

### ⚡ Parallel Multi-LLM (yupp.ai style)
- Every question fires **all configured LLMs simultaneously** using `ThreadPoolExecutor`
- Total time ≈ slowest single LLM instead of sum of all
- Answers appear as equal-width cards side-by-side
- Each card shows the LLM name, response time, and its answer
- Failed LLMs show an error card without blocking the others

### 🧠 Chat Memory
- Sliding window of last 6 conversation turns (configurable)
- History injected into both the query rewriter and the LLM prompt
- Follow-up questions work naturally: "what about its cost?" understands the prior context
- Memory cleared with a single button click

### 📚 Source Citations
- Every answer shows which documents and pages were used
- Similarity score and rerank score displayed per source
- Visual score bar for quick relevance scanning
- Full passage text in expandable source panel

### 📐 Retrieval Metrics
Every question shows:
- Rewrite time, retrieval time, rerank time, LLM response time, total time
- Number of candidates fetched vs kept after reranking

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
pip install -r requirements_v2.txt
```

### 4 — Start Endee (Docker)

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

Verify at **http://localhost:8080**

### 5 — Configure environment

```bash
cp .env.example .env
```

Minimum required in `.env`:

```env
GEMINI_API_KEY=your_gemini_key_here
LLM_PROVIDER=gemini
MULTI_LLM_PROVIDERS=gemini,anthropic
```

> 💡 Get a free Gemini key at https://makersuite.google.com/app/apikey
> 💡 Use `LLM_PROVIDER=mock` and `MULTI_LLM_PROVIDERS=mock` to test with no API keys at all.

### 6 — Launch AskMyNotes

```bash
streamlit run app_v2.py
```

Open **http://localhost:8501** — you'll see the upload screen. Drop in a PDF and start asking questions.

---

## 🖥 UI Flow

```
① Upload screen (first visit)
   ┌────────────────────────────────┐
   │  🧠 AskMyNotes                 │
   │                                │
   │  Upload notes. Ask questions.  │
   │  Get AI answers.               │
   │                                │
   │  ┌── Drop PDF here ──────────┐ │
   │  │  📄 notes.pdf  ✅ indexed  │ │
   │  └───────────────────────────┘ │
   │                                │
   │  ⚡ Start asking questions →   │
   └────────────────────────────────┘
              ↓ click button
② Chat arena
   ┌─────────────────────────────────────────────────────────┐
   │  Q: What is HNSW indexing?                              │
   │  ┌─ Gemini ─┐  ┌─ Claude ─┐  ┌─ GPT-4o ─┐  ┌─Ollama─┐│
   │  │ HNSW is… │  │ HNSW is… │  │ HNSW is… │  │HNSW is…││
   │  └──────────┘  └──────────┘  └──────────┘  └────────┘│
   │  ✏️ rewrite 340ms  🔍 retrieve 38ms  📊 rerank 122ms   │
   │  📚 5 sources ▼                                        │
   └─────────────────────────────────────────────────────────┘
```

---

## 🤖 LLM Providers

| Provider | `MULTI_LLM_PROVIDERS` value | API Key needed |
|----------|---------------------------|----------------|
| Google Gemini | `gemini` | `GEMINI_API_KEY` — [get free key](https://makersuite.google.com/app/apikey) |
| Anthropic Claude | `anthropic` | `ANTHROPIC_API_KEY` |
| OpenAI GPT-4o | `openai` | `OPENAI_API_KEY` |
| Ollama (local) | `ollama` | None — [install Ollama](https://ollama.com) |
| Mock (testing) | `mock` | None |

**Example — run only Gemini + Claude:**
```env
MULTI_LLM_PROVIDERS=gemini,anthropic
```

**Example — fully offline with Ollama:**
```env
MULTI_LLM_PROVIDERS=ollama,mock
OLLAMA_MODEL=llama3
```

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
| `MULTI_LLM_TIMEOUT` | `60` | Seconds before LLM times out |
| `MEMORY_MAX_TURNS` | `6` | Conversation turns kept in memory |

---

## 🧩 How Each Module Works

### `pdf_loader.py`
Accepts a PDF as bytes (from Streamlit uploader), extracts text page-by-page with `pdfplumber`, runs the identical chunking and embedding pipeline as `embed_documents.py`, and upserts vectors into Endee with `{filename, page, source="pdf"}` metadata. The existing retrieval pipeline is completely unaware — it just sees more vectors.

### `query_rewriter.py`
Uses Gemini to transform the user's raw question before it hits Endee. Three strategies: `standalone` (uses chat history to resolve "it" / "that"), `expand` (adds keywords), `hyde` (generates a hypothetical answer whose embedding is closer to real document passages). Falls back to the original query if Gemini is unavailable.

### `reranker.py`
Wraps `sentence_transformers.CrossEncoder`. Takes the query and all retrieved chunks, scores each `(query, chunk)` pair together (not independently like the bi-encoder), and returns only the top-K by score. Adds a `rerank_score` key to each chunk dict — the rest of the pipeline is unchanged.

### `llm_clients.py`
Houses all LLM backends (Gemini, Claude, GPT-4o, Ollama, Mock) under a single `call_llm(prompt, provider)` router. Also provides `call_llm_parallel(prompt, providers)` which uses `ThreadPoolExecutor` to fire all providers simultaneously — total wall time equals the slowest single LLM.

### `chat_memory.py`
Stores the last N conversation turns in memory. Provides `as_list()` for the query rewriter and `build_prompt_with_memory()` which injects prior exchanges into the LLM prompt so follow-up questions work naturally.

### `enhanced_rag_pipeline.py`
Orchestrates all v2 modules. Wraps the original `RAGPipeline` without modifying it. `run()` handles single-LLM queries. `run_parallel()` does one shared retrieval pass then fires all LLMs in parallel.

### `app_v2.py` — AskMyNotes
Two-screen Streamlit app. **Screen 1** (upload gate): centered PDF uploader, progress bar, "Start asking" button — chat input is hidden until at least one PDF is indexed. **Screen 2** (chat arena): yupp.ai-style parallel answer cards, rewrite banner, metrics chips, source expander.

---

## 🔵 How Endee Is Used

Endee is the retrieval engine powering every answer. Three interactions happen:

**1. Index creation** (once):
```python
from endee import Endee, Precision
client = Endee()
client.set_base_url("http://localhost:8080/api/v1")
client.create_index(name="rag_documents", dimension=384,
                    space_type="cosine", precision=Precision.INT8)
```

**2. Vector upsert** (on every PDF upload):
```python
index = client.get_index(name="rag_documents")
index.upsert([{
    "id":     "pdf_a1b2_p0001_c000",
    "vector": [0.12, -0.34, ...],    # 384 floats
    "meta":   {"text": "...", "title": "notes.pdf — Page 1",
               "filename": "notes.pdf", "page": 1},
    "filter": {"source": "pdf"},
}])
```

**3. Similarity query** (on every question):
```python
results = index.query(vector=query_embedding, top_k=15, ef=128)
# Returns the 15 most semantically similar chunks in < 5ms
```

---

## 🐛 Troubleshooting

| Problem | Fix |
|---------|-----|
| Cannot connect to Endee | `docker compose up -d` · verify at http://localhost:8080 |
| `Index not found` | Run `python embed_documents.py` to create the index |
| PDF shows no text | PDF may be scanned/image-only — use a text-selectable PDF |
| Gemini API error | Check `GEMINI_API_KEY` in `.env` · get a free key at makersuite.google.com |
| LLM card shows error | That LLM is misconfigured or offline — other cards still work |
| Reranker slow on first query | Cross-encoder model downloads on first use (~50 MB) |

---

## 🚀 Potential Extensions

- [ ] OCR support for scanned PDFs (`pytesseract`)
- [ ] Image and table extraction from PDFs
- [ ] User authentication and per-user knowledge bases
- [ ] Cloud deployment of Endee (AWS / GCP / Azure)
- [ ] Streaming LLM responses (token-by-token in cards)
- [ ] Answer quality voting (thumbs up/down per LLM card)
- [ ] Export conversation as PDF report

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
