"""
config.py
─────────────────────────────────────────────────────────────────────────────
Centralised configuration for the RAG chatbot.
All tuneable parameters live here so they are easy to find and change.
─────────────────────────────────────────────────────────────────────────────
"""

import os
from dotenv import load_dotenv

# Load .env file if present (safe to call even if file does not exist)
load_dotenv()


# ── Endee Vector Database ─────────────────────────────────────────────────────
ENDEE_BASE_URL: str = os.getenv("ENDEE_BASE_URL", "http://localhost:8080/api/v1")
ENDEE_AUTH_TOKEN: str = os.getenv("ENDEE_AUTH_TOKEN", "")          # empty = no auth
ENDEE_INDEX_NAME: str = os.getenv("ENDEE_INDEX_NAME", "rag_documents")


# ── Embedding Model ───────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"   # HuggingFace model name
EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "384"))
EMBEDDING_DEVICE: str = "cpu"                # "cpu" or "cuda"


# ── Document Chunking ─────────────────────────────────────────────────────────
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "400"))       # max chars per chunk
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))  # overlap between chunks


# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "5"))   # docs returned per query
EF_SEARCH: int = 128                                         # Endee HNSW ef param


# ── LLM Provider ─────────────────────────────────────────────────────────────
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "anthropic")  # anthropic | openai | ollama | mock
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3")


# ── Data Paths ────────────────────────────────────────────────────────────────
DATA_DIR: str = "data"
DOCUMENTS_FILE: str = os.path.join(DATA_DIR, "documents.txt")
