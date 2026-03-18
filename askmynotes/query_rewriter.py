"""
query_rewriter.py
─────────────────────────────────────────────────────────────────────────────
QUERY REWRITING MODULE — new in v2

WHY QUERY REWRITING?
────────────────────
Users type conversational, ambiguous, or short queries:
  "tell me more"          ← needs memory context to be useful
  "what about the cost?"  ← unclear what "cost" refers to
  "RAG"                   ← too short for good retrieval

Rewriting transforms these into richer, self-contained queries that embed
much better in the 384-dim vector space, improving top-K recall from Endee.

ARCHITECTURE POSITION
─────────────────────
  User query
      │
      ▼
  query_rewriter.rewrite_query()   ← NEW — this module
      │
      ▼
  Improved query → SemanticSearch (unchanged)
      │
      ▼
  Endee similarity search (unchanged)

HOW IT WORKS
────────────
Three strategies, applied in order:

1. STANDALONE REWRITE  — makes follow-up questions self-contained
   "what about the cost?" + history → "What is the cost of RAG systems?"

2. KEYWORD EXPANSION   — adds synonyms and related terms
   "RAG" → "RAG retrieval augmented generation document question answering"

3. HYPOTHETICAL ANSWER — generates what an ideal answer would look like,
   then uses THAT as the query (HyDE technique). Hypothetical answers embed
   very close to real answers in the vector space.

Gemini is used because it's fast, cheap, and already integrated as the
primary LLM. The rewriter adds ~300–500ms but significantly improves recall.
─────────────────────────────────────────────────────────────────────────────
"""

import time
from typing import List, Dict, Optional

import config


# ─────────────────────────────────────────────────────────────────────────────
# GEMINI CLIENT (lazy-loaded)
# ─────────────────────────────────────────────────────────────────────────────

_gemini_model = None

def _get_gemini():
    """Lazy-load Gemini model once."""
    global _gemini_model
    if _gemini_model is not None:
        return _gemini_model

    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError(
            "google-generativeai not installed.\n"
            "Run: pip install google-generativeai"
        )

    if not config.GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY is not set in .env\n"
            "Get a free key at https://makersuite.google.com/app/apikey"
        )

    genai.configure(api_key=config.GEMINI_API_KEY)
    _gemini_model = genai.GenerativeModel(config.GEMINI_REWRITE_MODEL)
    return _gemini_model


def _call_gemini(prompt: str, temperature: float = 0.1) -> str:
    """Call Gemini with a prompt, return text response."""
    model = _get_gemini()
    import google.generativeai as genai
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=256,
        ),
    )
    return response.text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 1 — STANDALONE REWRITE
# Makes follow-up questions self-contained using conversation history
# ─────────────────────────────────────────────────────────────────────────────

STANDALONE_PROMPT = """You are a search query optimizer for a document retrieval system.

Given the conversation history and the user's latest question, rewrite the question 
as a clear, self-contained search query that captures the user's full intent.

Rules:
- Keep it concise (1-2 sentences maximum)
- Include important keywords from the conversation context
- Make it specific enough to retrieve precise document chunks
- Do NOT answer the question, only rewrite it
- If the question is already clear and self-contained, return it unchanged

Conversation history:
{history}

User's question: {query}

Rewritten search query (return ONLY the query, nothing else):"""


def rewrite_standalone(
    query: str,
    history: List[Dict[str, str]],
) -> str:
    """
    Make a follow-up question self-contained using conversation history.

    Example
    -------
    History: "Q: What is RAG? A: RAG stands for Retrieval-Augmented Generation..."
    Query:   "What are its main benefits?"
    Output:  "What are the main benefits of Retrieval-Augmented Generation (RAG)?"

    Parameters
    ----------
    query   : the user's current question
    history : list of {"role": "user"|"assistant", "content": str}
    """
    if not history:
        return query  # Nothing to contextualise against

    # Format last N turns for the prompt
    history_text = ""
    for turn in history[-4:]:  # Only last 4 turns to keep prompt short
        role = "User" if turn["role"] == "user" else "Assistant"
        content = turn["content"][:300]  # Truncate long answers
        history_text += f"{role}: {content}\n"

    prompt = STANDALONE_PROMPT.format(
        history=history_text.strip(),
        query=query,
    )

    try:
        return _call_gemini(prompt, temperature=0.1)
    except Exception:
        return query  # Graceful fallback — never break retrieval


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 2 — KEYWORD EXPANSION
# Adds related terms to improve recall for short / technical queries
# ─────────────────────────────────────────────────────────────────────────────

EXPANSION_PROMPT = """You are a search query expander for a technical document retrieval system.

Expand the following query by adding 3-5 relevant keywords, synonyms, and related 
technical terms that would help find relevant document passages.

Rules:
- Keep the original query at the start
- Add keywords separated by spaces after the original query
- Do NOT add sentences, just keywords
- Focus on technical synonyms and related concepts

Query: {query}

Expanded query (return ONLY the expanded query, nothing else):"""


def rewrite_expanded(query: str) -> str:
    """
    Expand a short query with related technical keywords.

    Example
    -------
    Input:  "RAG"
    Output: "RAG retrieval augmented generation document question answering
             vector search knowledge base LLM grounding"
    """
    # Only expand short queries — long ones are already specific enough
    if len(query.split()) >= 8:
        return query

    prompt = EXPANSION_PROMPT.format(query=query)

    try:
        return _call_gemini(prompt, temperature=0.2)
    except Exception:
        return query


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY 3 — HYPOTHETICAL DOCUMENT EMBEDDINGS (HyDE)
# Generates a hypothetical answer, uses that as the query vector
# This dramatically improves dense retrieval recall for factual questions
# ─────────────────────────────────────────────────────────────────────────────

HYDE_PROMPT = """You are a document generation assistant.

Write a short, factual passage (2-3 sentences) that would directly answer the 
following question. This passage will be used to search a document database.

Rules:
- Write as if it's content from a technical document or textbook
- Be specific and factual
- Do NOT say "According to..." or "Based on..." — just state the facts
- If you don't know the exact answer, write a plausible factual passage

Question: {query}

Hypothetical answer passage (2-3 sentences only):"""


def rewrite_hyde(query: str) -> str:
    """
    HyDE: Generate a hypothetical answer and use it as the retrieval query.

    WHY THIS WORKS
    ──────────────
    The embedding of a hypothetical answer is much closer to the embedding
    of real document passages than the embedding of the original question.
    Questions and answers live in different parts of the vector space.

    Example
    -------
    Query:  "What is cosine similarity?"
    HyDE:   "Cosine similarity is a metric that measures the angle between
             two vectors in high-dimensional space. It returns a value between
             -1 and 1, where 1 means identical direction..."
    The HyDE text embeds much closer to textbook explanations in Endee.
    """
    prompt = HYDE_PROMPT.format(query=query)

    try:
        return _call_gemini(prompt, temperature=0.3)
    except Exception:
        return query


# ─────────────────────────────────────────────────────────────────────────────
# MAIN REWRITE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def rewrite_query(
    query: str,
    history: Optional[List[Dict[str, str]]] = None,
    strategy: str = "standalone",
) -> Dict[str, str]:
    """
    Rewrite a user query to improve retrieval quality.

    Parameters
    ----------
    query    : original user question
    history  : conversation history [{"role": ..., "content": ...}]
    strategy : "standalone" | "expand" | "hyde" | "combined"
               - standalone : use history to make query self-contained
               - expand     : add related keywords
               - hyde       : generate hypothetical answer as query
               - combined   : standalone first, then expand result

    Returns
    -------
    dict:
        original    : str  — original query unchanged
        rewritten   : str  — improved query for retrieval
        strategy    : str  — which strategy was applied
        elapsed_ms  : int  — time taken in milliseconds
    """
    t0 = time.time()
    history = history or []
    original = query

    if strategy == "standalone":
        rewritten = rewrite_standalone(query, history)

    elif strategy == "expand":
        rewritten = rewrite_expanded(query)

    elif strategy == "hyde":
        rewritten = rewrite_hyde(query)

    elif strategy == "combined":
        # First make self-contained, then expand
        step1 = rewrite_standalone(query, history)
        rewritten = rewrite_expanded(step1)

    else:
        rewritten = query  # unknown strategy → passthrough

    return {
        "original":   original,
        "rewritten":  rewritten,
        "strategy":   strategy,
        "elapsed_ms": int((time.time() - t0) * 1000),
        "changed":    rewritten.strip() != original.strip(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_queries = [
        ("What is RAG?", [], "standalone"),
        ("tell me more about it", [
            {"role": "user", "content": "What is cosine similarity?"},
            {"role": "assistant", "content": "Cosine similarity measures the angle between two vectors..."},
        ], "standalone"),
        ("embeddings", [], "expand"),
        ("how does HNSW work?", [], "hyde"),
    ]

    for query, history, strategy in test_queries:
        result = rewrite_query(query, history, strategy)
        print(f"\nStrategy: {strategy}")
        print(f"  Original:  {result['original']}")
        print(f"  Rewritten: {result['rewritten']}")
        print(f"  Time:      {result['elapsed_ms']}ms")
