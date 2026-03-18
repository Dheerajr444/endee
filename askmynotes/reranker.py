"""
reranker.py
─────────────────────────────────────────────────────────────────────────────
CROSS-ENCODER RERANKING MODULE — new in v2

WHY RERANKING?
──────────────
The first-stage retrieval (Endee HNSW cosine similarity) uses a bi-encoder:
  • Query  → 384-dim vector
  • Chunk  → 384-dim vector
  • Score  = cosine(query_vec, chunk_vec)

Bi-encoders are fast but imprecise — the query and document are encoded
independently, so subtle relevance signals are lost.

A cross-encoder looks at query AND document together:
  • Input  = [query, chunk] concatenated
  • Score  = relevance score (0–1)

Cross-encoders are ~10x slower but 20-30% more accurate on relevance.

SOLUTION: Two-stage retrieval
  Stage 1: Endee fetches top-15 candidates quickly (bi-encoder)
  Stage 2: Cross-encoder scores all 15, keeps top-5

This gives the accuracy of a cross-encoder at near the speed of a bi-encoder.

ARCHITECTURE POSITION
─────────────────────
  Endee HNSW search → top-15 chunks   (fast, approximate)
          │
          ▼
  reranker.rerank()                    ← NEW — this module
          │
          ▼
  top-5 reranked chunks → LLM context  (accurate, final)
─────────────────────────────────────────────────────────────────────────────
"""

import time
from typing import List, Dict, Any, Optional

import config


# ─────────────────────────────────────────────────────────────────────────────
# CROSS-ENCODER MODEL (lazy-loaded)
# ─────────────────────────────────────────────────────────────────────────────

_cross_encoder = None

def _get_cross_encoder():
    """Lazy-load the cross-encoder model once."""
    global _cross_encoder
    if _cross_encoder is not None:
        return _cross_encoder

    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        raise ImportError(
            "sentence-transformers not installed (needed for CrossEncoder).\n"
            "Run: pip install sentence-transformers"
        )

    _cross_encoder = CrossEncoder(
        config.RERANKER_MODEL,
        max_length=512,
        device="cpu",
    )
    return _cross_encoder


# ─────────────────────────────────────────────────────────────────────────────
# RERANK FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def rerank(
    query: str,
    chunks: List[Dict[str, Any]],
    top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Rerank a list of retrieved chunks using a cross-encoder model.

    The input chunks come directly from SemanticSearch.search() and the
    output is a re-ordered subset in the same dict format — so the rest of
    the pipeline (build_context, LLM prompt) needs zero changes.

    Parameters
    ----------
    query  : user's question (or rewritten query)
    chunks : list of chunk dicts from SemanticSearch  (has "text", "similarity", etc.)
    top_k  : number of results to keep after reranking
             defaults to config.TOP_K_RESULTS

    Returns
    -------
    List of chunk dicts, re-sorted by cross-encoder relevance score.
    Each dict gets a new "rerank_score" key added.
    All other keys (text, title, similarity, page, filename…) preserved.
    """
    if not chunks:
        return chunks

    top_k = top_k or config.TOP_K_RESULTS

    # If fewer chunks than top_k, return as-is (nothing to rerank)
    if len(chunks) <= 1:
        return chunks[:top_k]

    t0 = time.time()

    # Build (query, passage) pairs for the cross-encoder
    pairs = [(query, chunk["text"]) for chunk in chunks]

    try:
        model = _get_cross_encoder()
        scores = model.predict(pairs, show_progress_bar=False)
    except Exception as exc:
        # If reranking fails for any reason, fall back to original order
        print(f"[reranker] Warning: cross-encoder failed ({exc}), "
              "falling back to cosine similarity order.")
        for chunk in chunks:
            chunk["rerank_score"] = chunk.get("similarity", 0.0)
        return chunks[:top_k]

    # Attach scores and re-sort
    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)

    reranked = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)
    elapsed = int((time.time() - t0) * 1000)

    return reranked[:top_k]


# ─────────────────────────────────────────────────────────────────────────────
# CLI TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Minimal smoke test with fake chunks
    test_chunks = [
        {"id": "c1", "text": "Vector databases store embeddings for similarity search.",
         "title": "Doc A", "similarity": 0.82, "chunk_idx": 0, "doc_id": "a"},
        {"id": "c2", "text": "Python is a popular programming language for data science.",
         "title": "Doc B", "similarity": 0.79, "chunk_idx": 0, "doc_id": "b"},
        {"id": "c3", "text": "Cosine similarity measures the angle between two vectors.",
         "title": "Doc C", "similarity": 0.75, "chunk_idx": 0, "doc_id": "c"},
    ]

    query = "How do vector databases work?"
    print(f"Query: {query}\n")
    print("Before reranking:")
    for i, c in enumerate(test_chunks, 1):
        print(f"  {i}. [{c['similarity']:.2f}] {c['text'][:60]}")

    reranked = rerank(query, test_chunks, top_k=3)

    print("\nAfter reranking:")
    for i, c in enumerate(reranked, 1):
        print(f"  {i}. [rerank={c['rerank_score']:.3f}] {c['text'][:60]}")
