# 🔍 RAG Document Chatbot — Powered by Endee Vector Database

A complete **Retrieval-Augmented Generation (RAG)** chatbot built with:

- **[Endee](https://endee.io)** — high-performance vector database
- **sentence-transformers** — `all-MiniLM-L6-v2` (384-dim embeddings)
- **Anthropic Claude / OpenAI / Ollama** — LLM generation
- **Streamlit** — web chat UI

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     INDEXING PIPELINE                           │
│  (run once with embed_documents.py)                             │
│                                                                 │
│  Documents → Chunking → all-MiniLM-L6-v2 → Endee Index         │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     QUERY PIPELINE                              │
│  (runs on every user question)                                  │
│                                                                 │
│  User Query                                                     │
│      │                                                          │
│      ▼                                                          │
│  Embed Query  ──── sentence-transformers (384-dim vector)       │
│      │                                                          │
│      ▼                                                          │
│  Vector Search ─── Endee HNSW cosine similarity (top-K)        │
│      │                                                          │
│      ▼                                                          │
│  Retrieved Chunks  (most semantically relevant passages)        │
│      │                                                          │
│      ▼                                                          │
│  Prompt Builder ── system + context + user question             │
│      │                                                          │
│      ▼                                                          │
│  LLM Generation ── Claude / GPT-4o / Ollama                    │
│      │                                                          │
│      ▼                                                          │
│  Final Answer  (grounded in retrieved facts)                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
project/
│
├── data/
│   └── documents.txt          ← Knowledge base (edit to add your own docs)
│
├── embed_documents.py         ← Step 1: Index documents into Endee
├── search_documents.py        ← Step 2: Semantic search demo
├── rag_pipeline.py            ← Step 3: Full RAG pipeline
├── chatbot.py                 ← Step 4: Terminal chatbot
├── app.py                     ← Step 5: Streamlit web UI
│
├── config.py                  ← All settings in one place
├── requirements.txt           ← Python dependencies
└── .env.example               ← Environment variable template
```

---

## Quick Start

### Prerequisites

- Python ≥ 3.8
- Docker Desktop (Windows/Mac) or Docker Engine (Linux)
- Git

### 1. Clone & enter the project

```bash
git clone https://github.com/your-username/rag-endee-chatbot.git
cd rag-endee-chatbot
```

### 2. Create and activate a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start Endee with Docker

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

Then start it:

```bash
docker compose up -d
```

Verify at [http://localhost:8080](http://localhost:8080)

### 5. Configure your environment

```bash
cp .env.example .env
# Edit .env and set your ANTHROPIC_API_KEY (or choose another LLM_PROVIDER)
```

### 6. Embed documents into Endee

```bash
python embed_documents.py
```

Output:
```
Step 1 / 5 – Loading documents   → 8 documents loaded
Step 2 / 5 – Chunking documents  → 42 chunks created
Step 3 / 5 – Loading model       → all-MiniLM-L6-v2 ready
Step 4 / 5 – Generating embeddings → 42 × 384-dim vectors
Step 5 / 5 – Inserting into Endee → 42 vectors upserted
```

### 7. Test semantic search

```bash
python search_documents.py
```

### 8. Run the terminal chatbot

```bash
python chatbot.py
```

### 9. Launch the Streamlit UI

```bash
streamlit run app.py
```

Visit [http://localhost:8501](http://localhost:8501)

---

## Key Concepts

### What is a Vector Database?

A vector database stores **high-dimensional numeric arrays** (vectors) rather than structured rows. Its killer feature is **similarity search** — given a query vector, find the K stored vectors most similar to it in O(log n) time using algorithms like HNSW.

Traditional SQL: `WHERE name = 'Python'` (exact match)
Vector DB: "find documents semantically similar to 'scripting language'"

### What are Embeddings?

An embedding is a **fixed-size numeric representation** of text that captures semantic meaning. The `all-MiniLM-L6-v2` model maps any sentence into a 384-dimensional vector where:
- "How does Docker work?" → `[0.12, -0.34, 0.07, …]` (384 numbers)
- "What is container technology?" → `[0.11, -0.31, 0.09, …]` (very similar!)
- "What is quantum physics?" → `[0.81, 0.22, -0.45, …]` (very different!)

### What is Cosine Similarity?

Cosine similarity measures the **angle** between two vectors:

```
similarity = cos(θ) = (A · B) / (|A| × |B|)
```

- `1.0` = identical direction (most similar)
- `0.0` = perpendicular (unrelated)
- `-1.0` = opposite (antonyms)

After L2-normalising embeddings, cosine similarity equals the dot product, making it computationally cheap.

### What is RAG?

**Retrieval-Augmented Generation** solves a fundamental LLM problem: models hallucinate when asked about facts they don't know. RAG gives the model a "cheat sheet":

1. **Retrieve** relevant passages from a knowledge base using vector search
2. **Augment** the prompt with those passages as grounded context
3. **Generate** a response that must be based on the provided context

Benefits:
- ✅ Grounded in real documents (fewer hallucinations)
- ✅ Knowledge base is easy to update (no model retraining)
- ✅ Can cite sources
- ✅ Works with any LLM

### Where does Endee Fit?

Endee is the **vector storage and retrieval layer**:

| Component | Role |
|-----------|------|
| sentence-transformers | Convert text to vectors |
| **Endee** | **Store vectors, serve similarity queries** |
| Anthropic/OpenAI | Generate natural language answers |

Endee handles the hardest part: finding the most similar vectors among thousands or millions at sub-millisecond latency using HNSW indexing.

---

## LLM Providers

Set `LLM_PROVIDER` in your `.env`:

| Value | Requires | Notes |
|-------|----------|-------|
| `anthropic` | `ANTHROPIC_API_KEY` | Claude – default, best quality |
| `openai` | `OPENAI_API_KEY` | GPT-4o |
| `ollama` | Local Ollama install | Fully offline, free |
| `mock` | Nothing | For testing without any API |

### Ollama Setup (offline / free)

```bash
# Install: https://ollama.com
ollama serve
ollama pull llama3
# Set LLM_PROVIDER=ollama in .env
```

---

## Customising the Knowledge Base

Edit `data/documents.txt` using the same format:

```
================================================================================
DOCUMENT N – YOUR DOCUMENT TITLE
================================================================================
Your document text here...

--------------------------------------------------------------------------------
```

Then re-run:
```bash
python embed_documents.py
```

Endee will update existing vectors automatically (upsert semantics).

---

## Debugging Tips

### Endee won't connect
```bash
docker ps                    # Is the container running?
docker logs endee-server     # Check server logs
curl http://localhost:8080   # Test HTTP connectivity
```

### Embeddings look wrong
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
vec = model.encode(["test sentence"])
print(vec.shape)   # Should be (1, 384)
```

### Search returns irrelevant results
- Increase `EF_SEARCH` in `config.py` for higher accuracy (slower)
- Re-embed with smaller `CHUNK_SIZE` for more precise chunks
- Check that documents were loaded correctly: `python embed_documents.py`

### LLM errors
- Set `LLM_PROVIDER=mock` in `.env` to test without API keys
- Check your API key is correctly set in `.env`

---

## Example Queries

```
What is a vector database and how does it differ from relational databases?
How does the RAG pipeline reduce hallucinations?
Explain how cosine similarity is calculated.
What model is used for embeddings in this project?
How do I run Endee using Docker?
What is the difference between machine learning and deep learning?
What are the key features of the Python programming language?
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Vector DB | Endee (open-source) |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` |
| LLM | Anthropic Claude / OpenAI GPT-4o / Ollama |
| Web UI | Streamlit |
| Terminal UI | Rich |
| Config | python-dotenv |

---

## License

MIT — use freely for learning, portfolios, and production systems.

---

*Built to demonstrate RAG architecture with Endee vector database.*
