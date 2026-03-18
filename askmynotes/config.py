"""
config.py — extended with Gemini, reranker, multi-LLM, memory settings.
All original lines preserved exactly. New settings appended at bottom.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Endee Vector Database ─────────────────────────────────────────────────────
ENDEE_BASE_URL: str = os.getenv("ENDEE_BASE_URL", "http://localhost:8080/api/v1")
ENDEE_AUTH_TOKEN: str = os.getenv("ENDEE_AUTH_TOKEN", "")
ENDEE_INDEX_NAME: str = os.getenv("ENDEE_INDEX_NAME", "rag_documents")

# ── Embedding Model ───────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "384"))
EMBEDDING_DEVICE: str = "cpu"

# ── Document Chunking ─────────────────────────────────────────────────────────
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "400"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "5"))
EF_SEARCH: int = 128

# ── LLM Provider ─────────────────────────────────────────────────────────────
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini")
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3")

# ── Data Paths ────────────────────────────────────────────────────────────────
DATA_DIR: str = "data"
DOCUMENTS_FILE: str = os.path.join(DATA_DIR, "documents.txt")

# =============================================================================
# NEW SETTINGS — added for v2 upgrade. Nothing above this line changed.
# =============================================================================

# ── Gemini ────────────────────────────────────────────────────────────────────
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
GEMINI_REWRITE_MODEL: str = os.getenv("GEMINI_REWRITE_MODEL", "gemini-1.5-flash")

# ── Reranker ─────────────────────────────────────────────────────────────────
RERANK_FETCH_K: int = int(os.getenv("RERANK_FETCH_K", "15"))
RERANKER_MODEL: str = os.getenv(
    "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# ── Multi-LLM parallel mode ───────────────────────────────────────────────────
MULTI_LLM_PROVIDERS: list = [
    p.strip()
    for p in os.getenv("MULTI_LLM_PROVIDERS", "gemini,anthropic").split(",")
    if p.strip()
]
MULTI_LLM_TIMEOUT: int = int(os.getenv("MULTI_LLM_TIMEOUT", "60"))

# ── Chat memory ───────────────────────────────────────────────────────────────
MEMORY_MAX_TURNS: int = int(os.getenv("MEMORY_MAX_TURNS", "6"))
