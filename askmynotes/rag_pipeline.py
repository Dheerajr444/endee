"""
rag_pipeline.py
─────────────────────────────────────────────────────────────────────────────
STEP 3 OF THE RAG PIPELINE – Retrieval + Generation

What this module does:
  1. Retrieves the most relevant document chunks from Endee (via SemanticSearch)
  2. Builds a grounded context block from the retrieved passages
  3. Constructs a well-engineered prompt that instructs the LLM to answer
     based ONLY on the provided context
  4. Sends the prompt to the configured LLM and returns the response

Supported LLM backends (set LLM_PROVIDER in .env):
  • anthropic  – Claude via Anthropic API (default)
  • openai     – GPT-4o / GPT-3.5 via OpenAI API
  • ollama     – Any local model via Ollama (llama3, mistral, etc.)
  • mock       – Returns a canned response (useful for offline testing)

─────────────────────────────────────────────────────────────────────────────
"""

import sys
import time
import textwrap
from typing import List, Dict, Any, Optional, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

import config
from search_documents import SemanticSearch

console = Console()


# ═════════════════════════════════════════════════════════════════════════════
# CONTEXT BUILDER
# ═════════════════════════════════════════════════════════════════════════════

def build_context(retrieved_chunks: List[Dict[str, Any]]) -> str:
    """
    Format retrieved chunks into a clean context block for the LLM prompt.

    WHY THIS MATTERS
    ────────────────
    The quality of the context block directly impacts LLM output quality.
    A well-structured context:
      • Numbers each passage so the LLM can cite sources.
      • Includes document title so the model can attribute information.
      • Shows similarity score so highly relevant chunks are salient.
      • Keeps formatting simple so the LLM spends tokens on reasoning,
        not parsing structure.

    Parameters
    ----------
    retrieved_chunks : list of dicts from SemanticSearch.search()

    Returns
    -------
    A formatted multi-line string ready to inject into the prompt.
    """
    if not retrieved_chunks:
        return "No relevant context found in the knowledge base."

    lines = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        lines.append(
            f"[{i}] SOURCE: {chunk['title']}  (relevance: {chunk['similarity']:.2f})\n"
            f"{chunk['text'].strip()}\n"
        )

    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
# PROMPT ENGINEERING
# ═════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a knowledgeable AI assistant with access to a curated knowledge base.

Your task:
1. Answer the user's question using ONLY the context passages provided below.
2. If the context does not contain enough information to answer fully, say so clearly — do not fabricate facts.
3. Cite your sources by referring to passage numbers (e.g. "According to [1]…").
4. Be concise, accurate, and helpful.
5. If the question is outside the knowledge base, politely say you don't have that information available.
"""

def build_prompt(query: str, context: str) -> str:
    """
    Construct the full prompt injected into the LLM.

    PROMPT STRUCTURE
    ────────────────
    System role:  Instructions and persona
    Context:      Retrieved document chunks (the "R" in RAG)
    User query:   The question to answer

    This separation of system + context + query is the standard RAG prompt
    pattern. It grounds the model's answer in factual retrieved content while
    leveraging the LLM's language understanding and coherence.
    """
    return textwrap.dedent(f"""
    CONTEXT FROM KNOWLEDGE BASE:
    ─────────────────────────────
    {context}
    ─────────────────────────────

    USER QUESTION:
    {query}

    ANSWER:
    """).strip()


# ═════════════════════════════════════════════════════════════════════════════
# LLM BACKENDS
# ═════════════════════════════════════════════════════════════════════════════

def call_anthropic(prompt: str, system: str = SYSTEM_PROMPT) -> str:
    """Call Anthropic Claude API."""
    try:
        import anthropic
    except ImportError:
        console.print("[red]❌  anthropic SDK not installed: pip install anthropic[/red]")
        sys.exit(1)

    if not config.ANTHROPIC_API_KEY:
        raise ValueError(
            "ANTHROPIC_API_KEY is not set in .env  "
            "Get a key at https://console.anthropic.com"
        )

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def call_openai(prompt: str, system: str = SYSTEM_PROMPT) -> str:
    """Call OpenAI API (GPT-4o by default)."""
    try:
        from openai import OpenAI
    except ImportError:
        console.print("[red]❌  openai SDK not installed: pip install openai[/red]")
        sys.exit(1)

    if not config.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set in .env")

    client = OpenAI(api_key=config.OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        max_tokens=1024,
        temperature=0.2,
    )
    return response.choices[0].message.content


def call_ollama(prompt: str, system: str = SYSTEM_PROMPT) -> str:
    """Call a locally running Ollama model (no API key required)."""
    try:
        import requests
    except ImportError:
        console.print("[red]❌  requests not installed: pip install requests[/red]")
        sys.exit(1)

    url = f"{config.OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": config.OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }
    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["message"]["content"]
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            f"Cannot connect to Ollama at {config.OLLAMA_BASE_URL}.\n"
            f"Start it with: ollama serve\n"
            f"Then pull model: ollama pull {config.OLLAMA_MODEL}"
        )


def call_mock_llm(prompt: str, system: str = SYSTEM_PROMPT) -> str:
    """
    Mock LLM for offline testing – returns a templated response
    so you can test the full pipeline without any API keys.
    """
    context_start = prompt.find("CONTEXT FROM KNOWLEDGE BASE:")
    context_end = prompt.find("USER QUESTION:")
    context_excerpt = ""
    if context_start != -1 and context_end != -1:
        context_excerpt = prompt[context_start + 30:context_end].strip()[:200]

    return (
        "**[MOCK LLM RESPONSE – no real AI is running]**\n\n"
        "I found the following relevant context from your knowledge base:\n\n"
        f"{context_excerpt}...\n\n"
        "To enable real AI generation, set LLM_PROVIDER=anthropic in your "
        ".env file and add your ANTHROPIC_API_KEY."
    )


def call_llm(prompt: str) -> str:
    """
    Route to the appropriate LLM backend based on config.LLM_PROVIDER.

    Add more providers here by following the same pattern.
    """
    provider = config.LLM_PROVIDER.lower()

    if provider == "anthropic":
        return call_anthropic(prompt)
    elif provider == "openai":
        return call_openai(prompt)
    elif provider == "ollama":
        return call_ollama(prompt)
    elif provider == "mock":
        return call_mock_llm(prompt)
    else:
        raise ValueError(
            f"Unknown LLM provider: '{provider}'. "
            "Valid options: anthropic, openai, ollama, mock"
        )


# ═════════════════════════════════════════════════════════════════════════════
# RAG PIPELINE CLASS
# ═════════════════════════════════════════════════════════════════════════════

class RAGPipeline:
    """
    Full RAG pipeline: query → retrieve → build context → generate → respond.

    ARCHITECTURE RECAP
    ──────────────────

    User Query
        │
        ▼
    Embed Query  ──── (sentence-transformers all-MiniLM-L6-v2)
        │
        ▼
    Vector Search ─── (Endee HNSW cosine similarity search)
        │
        ▼
    Top-K Chunks ──── (most relevant document passages)
        │
        ▼
    Build Context ─── (formatted passages with source citations)
        │
        ▼
    LLM Prompt ─────── (system + context + question)
        │
        ▼
    LLM Generation ─── (Anthropic / OpenAI / Ollama)
        │
        ▼
    Final Answer
    """

    def __init__(self):
        self.searcher = SemanticSearch()

    def run(
        self,
        query: str,
        top_k: int = config.TOP_K_RESULTS,
        return_sources: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute the full RAG pipeline for a single query.

        Parameters
        ----------
        query          : user's natural language question
        top_k          : number of document chunks to retrieve
        return_sources : include retrieved chunks in the return dict

        Returns
        -------
        dict with keys:
          • answer          – the LLM-generated response string
          • sources         – list of retrieved chunk dicts (if return_sources)
          • context         – the formatted context string fed to the LLM
          • retrieval_time  – seconds taken for vector search
          • llm_time        – seconds taken for LLM generation
          • total_time      – total pipeline time in seconds
        """
        pipeline_start = time.time()

        # ── Step 1: Retrieve relevant chunks from Endee ───────────────────────
        t0 = time.time()
        retrieved = self.searcher.search(query, top_k=top_k)
        retrieval_time = time.time() - t0

        # ── Step 2: Build formatted context block ─────────────────────────────
        context = build_context(retrieved)

        # ── Step 3: Construct the LLM prompt ──────────────────────────────────
        prompt = build_prompt(query, context)

        # ── Step 4: Call the LLM ──────────────────────────────────────────────
        t0 = time.time()
        answer = call_llm(prompt)
        llm_time = time.time() - t0

        total_time = time.time() - pipeline_start

        result = {
            "query": query,
            "answer": answer,
            "context": context,
            "retrieval_time": round(retrieval_time, 3),
            "llm_time": round(llm_time, 3),
            "total_time": round(total_time, 3),
        }

        if return_sources:
            result["sources"] = retrieved

        return result


# ═════════════════════════════════════════════════════════════════════════════
# DISPLAY HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def display_rag_result(result: Dict[str, Any]) -> None:
    """Render a RAG result with Rich formatting."""
    # Answer panel
    console.print(Panel(
        Markdown(result["answer"]),
        title="[bold green]AI Answer[/bold green]",
        border_style="green",
        padding=(1, 2),
    ))

    # Source citations
    if result.get("sources"):
        console.print("\n[bold dim]Sources retrieved from Endee:[/bold dim]")
        for i, src in enumerate(result["sources"], 1):
            console.print(
                f"  [{i}] [cyan]{src['title']}[/cyan]  "
                f"[dim]similarity={src['similarity']:.3f}[/dim]"
            )

    # Timing summary
    console.print(
        f"\n[dim]⏱  Retrieval: {result['retrieval_time']*1000:.0f}ms  │  "
        f"LLM: {result['llm_time']*1000:.0f}ms  │  "
        f"Total: {result['total_time']*1000:.0f}ms[/dim]"
    )


# ═════════════════════════════════════════════════════════════════════════════
# DEMO / MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    console.print(Panel.fit(
        "[bold cyan]RAG Pipeline – End-to-End Demo[/bold cyan]\n"
        f"[dim]LLM Provider: {config.LLM_PROVIDER}[/dim]",
        border_style="cyan",
    ))

    pipeline = RAGPipeline()

    example_queries = [
        "What is Endee and why is it better than traditional databases?",
        "Explain how RAG reduces LLM hallucinations.",
        "How does cosine similarity relate to semantic search?",
    ]

    for query in example_queries:
        console.print(f"\n[bold yellow]Q: {query}[/bold yellow]")
        console.rule(style="dim")
        try:
            result = pipeline.run(query)
            display_rag_result(result)
        except Exception as exc:
            console.print(f"[red]Pipeline error: {exc}[/red]")
        console.print()


if __name__ == "__main__":
    main()
