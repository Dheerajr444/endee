"""
embed_documents.py
─────────────────────────────────────────────────────────────────────────────
STEP 1 OF THE RAG PIPELINE – Document Ingestion

What this script does:
  1. Loads raw text documents from data/documents.txt
  2. Splits them into overlapping chunks (to preserve context)
  3. Generates 384-dimensional embeddings with all-MiniLM-L6-v2
  4. Creates an Endee vector index (if it does not exist)
  5. Upserts each chunk + its embedding into Endee

Run this script once (or whenever documents change):
  python embed_documents.py

─────────────────────────────────────────────────────────────────────────────
"""

import sys
import time
import hashlib
from typing import List, Dict, Any

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.panel import Panel
from rich.table import Table

# --- Workaround for Endee SDK Bug ---
from pydantic import BaseModel
if not hasattr(BaseModel, 'get'):
    BaseModel.get = lambda self, key, default=None: getattr(self, key, default)
# -----

# ── Local imports ─────────────────────────────────────────────────────────────
import config

console = Console()


# ═════════════════════════════════════════════════════════════════════════════
# 1.  DOCUMENT LOADING
# ═════════════════════════════════════════════════════════════════════════════

def load_documents(filepath: str) -> List[Dict[str, str]]:
    """
    Parse a structured text file into individual named documents.

    The documents.txt format uses lines of '=' signs as separators and a
    header line starting with 'DOCUMENT N –' to name each document.

    Returns
    -------
    List of dicts: [{"id": str, "title": str, "text": str}, ...]
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            raw = f.read()
    except FileNotFoundError:
        console.print(f"[red]❌  File not found: {filepath}[/red]")
        sys.exit(1)

    documents = []
    # Split on the separator lines (80 '=' characters)
    sections = raw.split("=" * 80)

    for section in sections:
        section = section.strip()
        if not section:
            continue

        lines = section.strip().splitlines()
        # Find the title line (starts with 'DOCUMENT')
        title = "Unknown Document"
        text_lines = []
        for i, line in enumerate(lines):
            if line.startswith("DOCUMENT"):
                title = line.strip()
                text_lines = lines[i + 1:]
                break
        else:
            text_lines = lines

        # Strip separator dashes and blank lines, join remainder
        body = "\n".join(
            ln for ln in text_lines
            if not ln.startswith("-" * 10)
        ).strip()

        # if body:
        #     doc_id = hashlib.md5(title.encode()).hexdigest()[:8]
        #     documents.append({"id": doc_id, "title": title, "text": body})
        if body:
            # Hash the title AND the body text to guarantee a unique ID
            unique_string = f"{title}_{body}"
            doc_id = hashlib.md5(unique_string.encode()).hexdigest()[:8]
            documents.append({"id": doc_id, "title": title, "text": body})

    return documents


# ═════════════════════════════════════════════════════════════════════════════
# 2.  DOCUMENT CHUNKING
# ═════════════════════════════════════════════════════════════════════════════

def chunk_text(
    text: str,
    chunk_size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
) -> List[str]:
    """
    Split text into overlapping chunks.

    WHY CHUNKING?
    LLMs and embedding models have context-length limits. Chunking ensures:
      • Each vector represents a focused, coherent passage.
      • Overlap prevents information loss at chunk boundaries.
      • Smaller chunks = faster retrieval & more precise similarity matching.

    Strategy: word-boundary-aware sliding window.
    """
    words = text.split()
    chunks = []

    # Estimate words per chunk (average English word ≈ 5 chars)
    words_per_chunk = max(1, chunk_size // 5)
    step = max(1, words_per_chunk - overlap // 5)

    start = 0
    while start < len(words):
        end = start + words_per_chunk
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += step

    return chunks


def create_chunks_from_documents(
    documents: List[Dict[str, str]]
) -> List[Dict[str, Any]]:
    """
    Convert raw documents into chunk records ready for embedding.

    Each chunk record contains:
      • id        – unique identifier for the chunk
      • text      – the chunk text
      • doc_id    – the parent document's id
      • title     – the parent document's title
      • chunk_idx – position of this chunk within its document
    """
    all_chunks = []

    for doc in documents:
        chunks = chunk_text(doc["text"])
        for idx, chunk in enumerate(chunks):
            # Stable, reproducible ID based on doc + position
            chunk_id = f"{doc['id']}_chunk{idx:03d}"
            all_chunks.append({
                "id": chunk_id,
                "text": chunk,
                "doc_id": doc["id"],
                "title": doc["title"],
                "chunk_idx": idx,
            })

    return all_chunks


# ═════════════════════════════════════════════════════════════════════════════
# 3.  EMBEDDING GENERATION
# ═════════════════════════════════════════════════════════════════════════════

def load_embedding_model():
    """
    Load the sentence-transformers model.

    WHY all-MiniLM-L6-v2?
      • Compact (22 MB) yet highly accurate for semantic similarity.
      • 384-dimensional output – good balance of quality vs. storage.
      • Fastest inference among models with similar recall benchmarks.
      • No API key required – runs entirely offline.
    """
    console.print(f"[cyan]⚙  Loading embedding model: {config.EMBEDDING_MODEL}[/cyan]")
    t0 = time.time()

    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(config.EMBEDDING_MODEL)
    except ImportError:
        console.print("[red]❌  sentence-transformers not installed.[/red]")
        console.print("    Run: [bold]pip install sentence-transformers[/bold]")
        sys.exit(1)

    elapsed = time.time() - t0
    console.print(f"[green]✓  Model loaded in {elapsed:.1f}s[/green]")
    return model


def generate_embeddings(
    model,
    texts: List[str],
    batch_size: int = 32,
) -> List[List[float]]:
    """
    Generate embedding vectors for a list of text strings.

    Parameters
    ----------
    model      : SentenceTransformer instance
    texts      : list of strings to embed
    batch_size : number of texts to encode at once (tune for GPU/CPU memory)

    Returns
    -------
    List of float lists, each of length EMBEDDING_DIMENSION (384).
    """
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,   # L2-normalise → cosine similarity = dot product
        convert_to_numpy=True,
    )
    return embeddings.tolist()


# ═════════════════════════════════════════════════════════════════════════════
# 4.  ENDEE VECTOR DATABASE OPERATIONS
# ═════════════════════════════════════════════════════════════════════════════

def get_endee_client():
    """Initialise and return a configured Endee client."""
    try:
        from endee import Endee
    except ImportError:
        console.print("[red]❌  endee SDK not installed.[/red]")
        console.print("    Run: [bold]pip install endee[/bold]")
        sys.exit(1)

    token = config.ENDEE_AUTH_TOKEN or None
    client = Endee(token) if token else Endee()
    client.set_base_url(config.ENDEE_BASE_URL)

    # Verify the server is reachable with a plain HTTP GET — avoids any SDK
    # response-parsing bugs that plague list_indexes().
    try:
        import requests as _req
        base = config.ENDEE_BASE_URL.rstrip("/")
        # Try the /indexes endpoint; fall back to root if that 404s
        resp = _req.get(base + "/indexes", timeout=5)
        if resp.status_code >= 500:
            raise ConnectionError(f"HTTP {resp.status_code}")
        console.print(f"[green]✓  Connected to Endee at {config.ENDEE_BASE_URL}[/green]")
    except Exception as exc:
        console.print(f"[red]❌  Cannot reach Endee server at {config.ENDEE_BASE_URL}[/red]")
        console.print(f"    Error: {exc}")
        console.print("\n[yellow]Troubleshooting tips:[/yellow]")
        console.print("  • Is Docker running?  docker ps")
        console.print("  • Start Endee:        docker compose up -d")
        console.print("  • Check logs:         docker logs endee-server")
        sys.exit(1)

    return client


def create_or_get_index(client):
    """
    Create the vector index in Endee if it does not already exist.

    Index settings:
      • dimension   = 384  (matches all-MiniLM-L6-v2 output)
      • space_type  = cosine  (cosine similarity for semantic search)
      • precision   = INT8  (quantised storage – faster with minor quality loss)
    """
    try:
        from endee import Precision
    except ImportError:
        console.print("[red]❌  Could not import Precision from endee SDK.[/red]")
        sys.exit(1)

    index_name = config.ENDEE_INDEX_NAME

    # ── Robust existence check ────────────────────────────────────────────────
    # list_indexes() return type varies across SDK versions (dict, list of
    # strings, list of dicts).  Rather than parsing the response, we simply
    # attempt to create the index and treat the "already exists" error as a
    # success signal — this works regardless of SDK version.
    console.print(f"[cyan]⚙  Creating index '{index_name}'…[/cyan]")
    try:
        client.create_index(
            name=index_name,
            dimension=config.EMBEDDING_DIMENSION,
            space_type="cosine",
            precision=Precision.INT8,
        )
        console.print(f"[green]✓  Index '{index_name}' created.[/green]")
    except Exception as exc:
        exc_str = str(exc).lower()
        # Common "already exists" signals from the Endee API
        if any(phrase in exc_str for phrase in ("already exists", "already_exists",
                                                  "conflict", "409", "duplicate")):
            console.print(f"[yellow]⚠  Index '{index_name}' already exists – reusing it.[/yellow]")
            console.print("   [dim]To rebuild from scratch, delete it via the Endee dashboard.[/dim]")
        else:
            console.print(f"[red]❌  Failed to create index '{index_name}': {exc}[/red]")
            raise

    return client.get_index(name=index_name)


def upsert_chunks(
    index,
    chunks: List[Dict[str, Any]],
    embeddings: List[List[float]],
    batch_size: int = 100,
) -> None:
    """
    Insert (or update) chunk vectors into the Endee index.

    Each vector record:
      • id     – unique chunk identifier
      • vector – 384-dim float list
      • meta   – arbitrary metadata returned with search results
      • filter – key-value pairs usable in filtered queries
    """
    total = len(chunks)
    inserted = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("Upserting vectors…", total=total)

        for start in range(0, total, batch_size):
            batch_chunks = chunks[start : start + batch_size]
            batch_embeddings = embeddings[start : start + batch_size]

            records = [
                {
                    "id": chunk["id"],
                    "vector": emb,
                    "meta": {
                        "text": chunk["text"],
                        "title": chunk["title"],
                        "doc_id": chunk["doc_id"],
                        "chunk_idx": chunk["chunk_idx"],
                    },
                    "filter": {
                        "doc_id": chunk["doc_id"],
                    },
                }
                for chunk, emb in zip(batch_chunks, batch_embeddings)
            ]

            try:
                index.upsert(records)
                inserted += len(records)
                progress.advance(task, len(records))
            except Exception as exc:
                console.print(f"\n[red]❌  Upsert failed for batch starting at {start}: {exc}[/red]")
                raise

    console.print(f"[green]✓  {inserted} chunks inserted into Endee.[/green]")


# ═════════════════════════════════════════════════════════════════════════════
# 5.  MAIN ENTRYPOINT
# ═════════════════════════════════════════════════════════════════════════════

def main():
    console.print(Panel.fit(
        "[bold cyan]RAG Chatbot – Document Embedding Pipeline[/bold cyan]\n"
        "[dim]Loading → Chunking → Embedding → Indexing in Endee[/dim]",
        border_style="cyan",
    ))

    # ── Step 1: Load documents ────────────────────────────────────────────────
    console.print("\n[bold]Step 1 / 5 – Loading documents[/bold]")
    documents = load_documents(config.DOCUMENTS_FILE)
    console.print(f"  Loaded [bold]{len(documents)}[/bold] documents from {config.DOCUMENTS_FILE}")

    # ── Step 2: Chunk documents ───────────────────────────────────────────────
    console.print("\n[bold]Step 2 / 5 – Chunking documents[/bold]")
    chunks = create_chunks_from_documents(documents)
    console.print(f"  Created [bold]{len(chunks)}[/bold] chunks "
                  f"(size≈{config.CHUNK_SIZE} chars, overlap={config.CHUNK_OVERLAP})")

    # Summary table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Document", style="cyan", no_wrap=True, max_width=50)
    table.add_column("Chunks", justify="right")
    chunk_counts: Dict[str, int] = {}
    for c in chunks:
        chunk_counts[c["title"]] = chunk_counts.get(c["title"], 0) + 1
    for title, count in chunk_counts.items():
        table.add_row(title[:50], str(count))
    console.print(table)

    # ── Step 3: Load embedding model ─────────────────────────────────────────
    console.print("\n[bold]Step 3 / 5 – Loading embedding model[/bold]")
    model = load_embedding_model()

    # ── Step 4: Generate embeddings ──────────────────────────────────────────
    console.print("\n[bold]Step 4 / 5 – Generating embeddings[/bold]")
    texts = [c["text"] for c in chunks]
    t0 = time.time()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Encoding {len(texts)} chunks…", total=None)
        embeddings = generate_embeddings(model, texts)
        progress.update(task, description="Encoding complete ✓")

    elapsed = time.time() - t0
    console.print(
        f"  Generated [bold]{len(embeddings)}[/bold] embeddings "
        f"({config.EMBEDDING_DIMENSION}-dim) in {elapsed:.1f}s"
    )
    console.print(f"  Shape: [{len(embeddings)} × {len(embeddings[0])}]")

    # ── Step 5: Upsert into Endee ─────────────────────────────────────────────
    console.print("\n[bold]Step 5 / 5 – Inserting vectors into Endee[/bold]")
    client = get_endee_client()
    index = create_or_get_index(client)
    upsert_chunks(index, chunks, embeddings)

    # ── Done ──────────────────────────────────────────────────────────────────
    console.print(Panel.fit(
        "[bold green]✅  Embedding pipeline complete![/bold green]\n\n"
        f"  Index:      [cyan]{config.ENDEE_INDEX_NAME}[/cyan]\n"
        f"  Documents:  {len(documents)}\n"
        f"  Chunks:     {len(chunks)}\n"
        f"  Dimensions: {config.EMBEDDING_DIMENSION}\n\n"
        "[dim]Next step → run: python search_documents.py[/dim]",
        border_style="green",
    ))


if __name__ == "__main__":
    main()
