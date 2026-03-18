"""
enhanced_rag_pipeline.py
─────────────────────────────────────────────────────────────────────────────
UPGRADED RAG PIPELINE — v2 extension

The original rag_pipeline.py is NOT modified.
This module wraps and extends it with:
  • Query rewriting     (Gemini rewrites ambiguous questions)
  • Cross-encoder reranking  (improves passage relevance before LLM)
  • Chat memory         (multi-turn conversation support)
  • Source citations    (filename, page number, similarity, rerank score)
  • Retrieval metrics   (timing for each pipeline stage)
  • Parallel multi-LLM  (run all LLMs simultaneously, show side-by-side)

ARCHITECTURE
────────────
  User question
      │
      ▼
  ChatMemory.as_list()  →  query_rewriter.rewrite_query()
      │                     (standalone / expand / hyde)
      ▼
  Rewritten query
      │
      ▼
  SemanticSearch.search(top_k=RERANK_FETCH_K)   ← fetches 15, not 5
      │   (Endee HNSW cosine — unchanged)
      ▼
  reranker.rerank(top_k=TOP_K_RESULTS)          ← keeps best 5
      │   (cross-encoder — NEW)
      ▼
  build_context()  →  ChatMemory.build_prompt_with_memory()
      │
      ▼
  call_llm()  OR  call_llm_parallel()
      │
      ▼
  Answer + sources + metrics
      │
      ▼
  ChatMemory.add_turn()
─────────────────────────────────────────────────────────────────────────────
"""

import time
from typing import Dict, Any, List, Optional

import config

# ── Import existing modules unchanged ────────────────────────────────────────
from search_documents import SemanticSearch
from rag_pipeline import build_context        # reuse — not duplicated
from llm_clients import call_llm, call_llm_parallel

# ── Import new v2 modules ─────────────────────────────────────────────────────
from reranker import rerank
from query_rewriter import rewrite_query
from chat_memory import ChatMemory


class EnhancedRAGPipeline:
    """
    Drop-in upgrade for RAGPipeline with all v2 features.

    Can replace RAGPipeline in the Streamlit UI by swapping one import line.
    The original RAGPipeline in rag_pipeline.py is untouched and still works.

    Features controlled by constructor flags so you can disable any feature
    for debugging or A/B testing without code changes.
    """

    def __init__(
        self,
        enable_rewriting: bool = True,
        rewrite_strategy: str = "standalone",
        enable_reranking: bool = True,
        enable_memory: bool = True,
    ):
        self.searcher = SemanticSearch()
        self.memory = ChatMemory()

        self.enable_rewriting = enable_rewriting
        self.rewrite_strategy = rewrite_strategy
        self.enable_reranking = enable_reranking
        self.enable_memory = enable_memory

    # ── Single-LLM run ────────────────────────────────────────────────────────

    def run(
        self,
        query: str,
        top_k: int = None,
        return_sources: bool = True,
        provider: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute the full enhanced RAG pipeline for one query.

        Returns
        -------
        dict with keys:
          answer           : str   — LLM response
          sources          : list  — retrieved + reranked chunks with metadata
          original_query   : str   — user's original question
          rewritten_query  : str   — query after rewriting (may be same)
          rewrite_strategy : str   — which rewrite strategy was used
          context          : str   — formatted context fed to LLM
          rewrite_time_ms  : int
          retrieval_time_ms: int
          rerank_time_ms   : int
          llm_time_ms      : int
          total_time_ms    : int
          provider         : str   — which LLM was used
        """
        top_k = top_k or config.TOP_K_RESULTS
        t_total = time.time()
        metrics = {}

        # ── Step 1: Query rewriting ───────────────────────────────────────────
        t0 = time.time()
        if self.enable_rewriting:
            rewrite_result = rewrite_query(
                query=query,
                history=self.memory.as_list() if self.enable_memory else [],
                strategy=self.rewrite_strategy,
            )
            retrieval_query = rewrite_result["rewritten"]
        else:
            rewrite_result = {"original": query, "rewritten": query,
                               "strategy": "none", "changed": False}
            retrieval_query = query
        metrics["rewrite_time_ms"] = int((time.time() - t0) * 1000)

        # ── Step 2: Retrieve from Endee (fetch more for reranking) ────────────
        t0 = time.time()
        fetch_k = config.RERANK_FETCH_K if self.enable_reranking else top_k
        retrieved = self.searcher.search(retrieval_query, top_k=fetch_k)
        metrics["retrieval_time_ms"] = int((time.time() - t0) * 1000)
        metrics["retrieved_count"] = len(retrieved)

        # ── Step 3: Rerank ────────────────────────────────────────────────────
        t0 = time.time()
        if self.enable_reranking and len(retrieved) > 1:
            final_chunks = rerank(retrieval_query, retrieved, top_k=top_k)
        else:
            final_chunks = retrieved[:top_k]
        metrics["rerank_time_ms"] = int((time.time() - t0) * 1000)

        # ── Step 4: Build context ─────────────────────────────────────────────
        context = build_context(final_chunks)

        # ── Step 5: Build prompt (with memory) ───────────────────────────────
        if self.enable_memory:
            prompt = self.memory.build_prompt_with_memory(
                base_prompt="", context=context, query=query
            )
        else:
            from rag_pipeline import build_prompt
            prompt = build_prompt(query, context)

        # ── Step 6: LLM generation ────────────────────────────────────────────
        t0 = time.time()
        answer = call_llm(prompt, provider=provider)
        metrics["llm_time_ms"] = int((time.time() - t0) * 1000)

        metrics["total_time_ms"] = int((time.time() - t_total) * 1000)

        # ── Step 7: Update memory ─────────────────────────────────────────────
        if self.enable_memory:
            self.memory.add_turn("user", query)
            self.memory.add_turn(
                "assistant", answer,
                metadata={"sources": [c.get("title") for c in final_chunks]}
            )

        result = {
            "query":            query,
            "original_query":   rewrite_result["original"],
            "rewritten_query":  rewrite_result["rewritten"],
            "rewrite_strategy": rewrite_result["strategy"],
            "answer":           answer,
            "context":          context,
            "provider":         provider or config.LLM_PROVIDER,
            # Backward-compat keys (original RAGPipeline used these)
            "retrieval_time":   metrics["retrieval_time_ms"] / 1000,
            "llm_time":         metrics["llm_time_ms"] / 1000,
            "total_time":       metrics["total_time_ms"] / 1000,
            **metrics,
        }

        if return_sources:
            result["sources"] = final_chunks

        return result

    # ── Parallel multi-LLM run ────────────────────────────────────────────────

    def run_parallel(
        self,
        query: str,
        top_k: int = None,
        providers: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run retrieval once, then send the same prompt to multiple LLMs in
        parallel. Returns all answers for side-by-side comparison in the UI.

        Returns
        -------
        dict with keys:
          sources          : list of reranked chunks
          context          : str — context block
          original_query   : str
          rewritten_query  : str
          llm_results      : list of {provider, answer, success, error, elapsed_ms}
          retrieval_time_ms: int
          rerank_time_ms   : int
          total_time_ms    : int
        """
        top_k = top_k or config.TOP_K_RESULTS
        providers = providers or config.MULTI_LLM_PROVIDERS
        t_total = time.time()

        # ── Rewrite + Retrieve + Rerank (once, shared by all LLMs) ───────────
        t0 = time.time()
        if self.enable_rewriting:
            rw = rewrite_query(
                query=query,
                history=self.memory.as_list() if self.enable_memory else [],
                strategy=self.rewrite_strategy,
            )
            retrieval_query = rw["rewritten"]
        else:
            rw = {"original": query, "rewritten": query, "strategy": "none"}
            retrieval_query = query
        rewrite_ms = int((time.time() - t0) * 1000)

        t0 = time.time()
        fetch_k = config.RERANK_FETCH_K if self.enable_reranking else top_k
        retrieved = self.searcher.search(retrieval_query, top_k=fetch_k)
        retrieval_ms = int((time.time() - t0) * 1000)

        t0 = time.time()
        if self.enable_reranking and len(retrieved) > 1:
            final_chunks = rerank(retrieval_query, retrieved, top_k=top_k)
        else:
            final_chunks = retrieved[:top_k]
        rerank_ms = int((time.time() - t0) * 1000)

        context = build_context(final_chunks)

        # Build prompt with memory
        if self.enable_memory:
            prompt = self.memory.build_prompt_with_memory(
                base_prompt="", context=context, query=query
            )
        else:
            from rag_pipeline import build_prompt
            prompt = build_prompt(query, context)

        # ── Call all LLMs in parallel ─────────────────────────────────────────
        llm_results = call_llm_parallel(prompt, providers=providers)

        # ── Update memory with the fastest successful answer ──────────────────
        if self.enable_memory:
            best = next((r for r in llm_results if r["success"]), None)
            if best:
                self.memory.add_turn("user", query)
                self.memory.add_turn("assistant", best["answer"])

        return {
            "query":             query,
            "original_query":    rw["original"],
            "rewritten_query":   rw["rewritten"],
            "sources":           final_chunks,
            "context":           context,
            "llm_results":       llm_results,
            "rewrite_time_ms":   rewrite_ms,
            "retrieval_time_ms": retrieval_ms,
            "rerank_time_ms":    rerank_ms,
            "total_time_ms":     int((time.time() - t_total) * 1000),
        }

    # ── Memory management ─────────────────────────────────────────────────────

    def clear_memory(self):
        self.memory.clear()

    def get_memory_turns(self) -> List[Dict]:
        return self.memory.to_display_list()
