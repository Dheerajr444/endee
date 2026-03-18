"""
search_documents.py
─────────────────────────────────────────────────────────────────────────────
STEP 2 OF THE RAG PIPELINE – Semantic Similarity Search

What this script does:
  1. Encodes a user query using the same embedding model used for indexing.
  2. Sends the query vector to Endee for approximate nearest-neighbour (ANN)
     search using HNSW indexing.
  3. Returns the top-k most semantically similar document chunks.

Can be run interactively:
  python search_documents.py

Or imported by other modules:
  from search_documents import SemanticSearch
  searcher = SemanticSearch()
  results = searcher.search("What is cosine similarity?")

─────────────────────────────────────────────────────────────────────────────
"""

import sys
import time
from typing import List, Dict, Any, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

import config

console = Console()


# ═════════════════════════════════════════════════════════════════════════════
# SEMANTIC SEARCH CLASS
# ═════════════════════════════════════════════════════════════════════════════

class SemanticSearch:
    """
    Wraps embedding + Endee vector search into a simple search interface.

    Usage
    -----
    searcher = SemanticSearch()
    results = searcher.search("How does RAG work?", top_k=5)

    Each result dict contains:
      • id          – chunk identifier
      • similarity  – cosine similarity score (0 – 1)
      • text        – the passage text
      • title       – source document title
      • doc_id      – parent document identifier
      • chunk_idx   – position within source document
    """

    def __init__(self):
        self._model = None       # lazy-loaded embedding model
        self._index = None       # lazy-loaded Endee index handle
        self._client = None

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_model(self):
        """Lazy-load the SentenceTransformer model once."""
        if self._model is not None:
            return
        console.print(f"[cyan]⚙  Loading embedding model ({config.EMBEDDING_MODEL})…[/cyan]")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            console.print("[red]❌  sentence-transformers not installed.[/red]")
            sys.exit(1)
        self._model = SentenceTransformer(config.EMBEDDING_MODEL)
        console.print("[green]✓  Model ready.[/green]")

    def _load_index(self):
        """Lazy-connect to Endee and get the index handle."""
        if self._index is not None:
            return
        try:
            from endee import Endee
        except ImportError:
            console.print("[red]❌  endee SDK not installed.[/red]")
            sys.exit(1)

        token = config.ENDEE_AUTH_TOKEN or None
        self._client = Endee(token) if token else Endee()
        self._client.set_base_url(config.ENDEE_BASE_URL)

        try:
            self._index = self._client.get_index(name=config.ENDEE_INDEX_NAME)
            console.print(
                f"[green]✓  Connected to Endee index '{config.ENDEE_INDEX_NAME}'.[/green]"
            )
        except Exception as exc:
            console.print(
                f"[red]❌  Could not access index '{config.ENDEE_INDEX_NAME}': {exc}[/red]"
            )
            console.print(
                "[yellow]Hint: Have you run embed_documents.py yet?[/yellow]"
            )
            sys.exit(1)

    def _embed_query(self, query: str) -> List[float]:
        """
        Encode a query string into a 384-dim embedding vector.

        IMPORTANT: We use the same model and normalisation settings as during
        indexing so that similarity scores are comparable.
        """
        self._load_model()
        vector = self._model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vector[0].tolist()

    # ── Public API ────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = config.TOP_K_RESULTS,
        doc_id_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic similarity search against the Endee index.

        HOW IT WORKS
        ────────────
        1. Query text is embedded into a 384-dim vector.
        2. Endee uses HNSW (Hierarchical Navigable Small World) graph search
           to find the 'top_k' vectors most similar to the query vector.
        3. Similarity is measured by cosine similarity (configured at index
           creation time).
        4. Results include the stored metadata so we can return the original
           text, not just vector IDs.

        HNSW EXPLAINED (in brief)
        ─────────────────────────
        HNSW builds a multi-layer graph where each node is a vector. Search
        starts at a random entry point in the top layer and greedily navigates
        toward the query vector, moving down layers for increasing precision.
        This gives O(log n) approximate search instead of O(n) brute force.

        Parameters
        ----------
        query          : natural language query string
        top_k          : number of results to return
        doc_id_filter  : optional – restrict results to one source document

        Returns
        -------
        List of result dicts sorted by similarity (highest first).
        """
        self._load_index()

        query_vector = self._embed_query(query)

        # Optional metadata filter
        filter_param = None
        if doc_id_filter:
            filter_param = [{"doc_id": {"$eq": doc_id_filter}}]

        try:
            raw_results = self._index.query(
                vector=query_vector,
                top_k=top_k,
                ef=config.EF_SEARCH,
                include_vectors=False,   # we don't need the raw vectors back
                filter=filter_param,
            )
        except Exception as exc:
            console.print(f"[red]❌  Endee query failed: {exc}[/red]")
            raise

        # Normalise the response format
        results = []
        for item in (raw_results or []):
            meta = item.get("meta") or {}
            results.append({
                "id": item.get("id", ""),
                "similarity": round(float(item.get("similarity", 0.0)), 4),
                "text": meta.get("text", ""),
                "title": meta.get("title", ""),
                "doc_id": meta.get("doc_id", ""),
                "chunk_idx": meta.get("chunk_idx", 0),
            })

        return results

    def describe_index(self) -> Dict[str, Any]:
        """Return index metadata from Endee (vector count, dimensions, etc.)."""
        self._load_index()
        try:
            return self._index.describe() or {}
        except Exception as exc:
            return {"error": str(exc)}


# ═════════════════════════════════════════════════════════════════════════════
# DISPLAY HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def display_results(query: str, results: List[Dict[str, Any]]) -> None:
    """Pretty-print search results in a Rich table."""
    console.print(
        f"\n[bold cyan]Query:[/bold cyan] {query}\n"
        f"[dim]Returned {len(results)} result(s)[/dim]\n"
    )

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    for rank, r in enumerate(results, 1):
        score_color = "green" if r["similarity"] > 0.7 else (
            "yellow" if r["similarity"] > 0.4 else "red"
        )
        score_bar = "█" * int(r["similarity"] * 20)

        panel_content = (
            f"[bold]{r['title']}[/bold]  "
            f"[dim](chunk #{r['chunk_idx']})[/dim]\n\n"
            f"{r['text'][:300]}{'…' if len(r['text']) > 300 else ''}\n\n"
            f"Score: [{score_color}]{r['similarity']:.4f}  {score_bar}[/{score_color}]"
        )
        console.print(Panel(
            panel_content,
            title=f"[bold]Result #{rank}[/bold]",
            border_style=score_color,
            padding=(0, 1),
        ))


# ═════════════════════════════════════════════════════════════════════════════
# INTERACTIVE DEMO
# ═════════════════════════════════════════════════════════════════════════════

EXAMPLE_QUERIES = [
    "What is a vector database?",
    "How does cosine similarity work?",
    "Explain the RAG pipeline",
    "What is the all-MiniLM-L6-v2 model?",
    "How do I run Endee with Docker?",
    "What are large language models?",
]


def interactive_search(searcher: SemanticSearch) -> None:
    """Run an interactive search loop in the terminal."""
    console.print(Panel.fit(
        "[bold cyan]Semantic Search – Interactive Mode[/bold cyan]\n"
        "[dim]Type a question to search the knowledge base.\n"
        "Type 'examples' for sample queries, 'quit' to exit.[/dim]",
        border_style="cyan",
    ))

    # Show index stats
    info = searcher.describe_index()
    if info and "error" not in info:
        console.print(f"\n[dim]Index: {config.ENDEE_INDEX_NAME} | "
                      f"Vectors: {info.get('count', '?')} | "
                      f"Dimensions: {info.get('dimension', config.EMBEDDING_DIMENSION)}[/dim]\n")

    while True:
        try:
            query = console.input("[bold green]Search >[/bold green] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            console.print("[dim]Goodbye![/dim]")
            break
        if query.lower() == "examples":
            console.print("\n[bold]Example queries:[/bold]")
            for i, q in enumerate(EXAMPLE_QUERIES, 1):
                console.print(f"  {i}. {q}")
            console.print()
            continue

        t0 = time.time()
        results = searcher.search(query)
        elapsed = time.time() - t0

        display_results(query, results)
        console.print(f"[dim]Search time: {elapsed*1000:.1f} ms[/dim]\n")


def main():
    searcher = SemanticSearch()

    # Run example queries first to show the system works, then drop into
    # interactive mode.
    console.print(Panel.fit(
        "[bold]Running example searches…[/bold]",
        border_style="magenta",
    ))

    demo_queries = [
        "What is a vector database?",
        "How does RAG reduce hallucinations?",
    ]

    for q in demo_queries:
        t0 = time.time()
        results = searcher.search(q, top_k=3)
        elapsed = time.time() - t0
        display_results(q, results)
        console.print(f"[dim]Search time: {elapsed*1000:.1f} ms[/dim]\n")

    # Drop into interactive mode
    interactive_search(searcher)


if __name__ == "__main__":
    main()
