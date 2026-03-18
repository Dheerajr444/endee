"""
llm_clients.py
─────────────────────────────────────────────────────────────────────────────
LLM CLIENTS MODULE — new in v2

Contains:
  1. Individual LLM callers  — Gemini, Claude, GPT, Ollama, Mock
  2. call_llm()              — single-provider router (used by rag_pipeline.py)
  3. call_llm_parallel()     — runs multiple LLMs simultaneously using threads
                               and returns all answers side-by-side

DESIGN PRINCIPLE
────────────────
rag_pipeline.py already imports call_llm() from this file.
We keep that function signature IDENTICAL so rag_pipeline.py needs zero edits.
call_llm_parallel() is purely additive — the UI calls it directly for the
multi-LLM comparison view.

PARALLEL EXECUTION
──────────────────
Python's GIL means CPU threads don't run truly in parallel, but these are
I/O-bound tasks (HTTP calls to LLM APIs) so ThreadPoolExecutor gives real
concurrency. Four LLMs that each take 3s run in ~3s total instead of 12s.
─────────────────────────────────────────────────────────────────────────────
"""

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from typing import Dict, List, Optional, Any

import config


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT (shared by all LLMs)
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a knowledgeable AI assistant with access to a curated knowledge base.

Your task:
1. Answer the user's question using ONLY the context passages provided below.
2. If the context does not contain enough information to answer fully, say so clearly — do not fabricate facts.
3. Cite your sources by referring to passage numbers (e.g. "According to [1]…").
4. Be concise, accurate, and helpful.
5. If the question is outside the knowledge base, politely say you don't have that information available.
"""


# ─────────────────────────────────────────────────────────────────────────────
# INDIVIDUAL LLM CALLERS
# ─────────────────────────────────────────────────────────────────────────────

def call_gemini(prompt: str, system: str = SYSTEM_PROMPT) -> str:
    """
    Call Google Gemini API.

    EXAMPLE USAGE (standalone):
        import google.generativeai as genai
        genai.configure(api_key="YOUR_KEY")
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content("What is RAG?")
        print(response.text)

    In this system, the full prompt already contains the context, so we
    prepend the system instruction to the user prompt.
    """
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError(
            "google-generativeai not installed.\n"
            "Run: pip install google-generativeai"
        )

    if not config.GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY not set in .env\n"
            "Get a free key at https://makersuite.google.com/app/apikey"
        )

    genai.configure(api_key=config.GEMINI_API_KEY)
    model = genai.GenerativeModel(
        model_name=config.GEMINI_MODEL,
        system_instruction=system,
    )
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            temperature=0.2,
            max_output_tokens=1024,
        ),
    )
    return response.text.strip()


def call_anthropic(prompt: str, system: str = SYSTEM_PROMPT) -> str:
    """Call Anthropic Claude API."""
    try:
        import anthropic
    except ImportError:
        raise ImportError("anthropic SDK not installed: pip install anthropic")

    if not config.ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY is not set in .env")

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
        raise ImportError("openai SDK not installed: pip install openai")

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
        raise ImportError("requests not installed: pip install requests")

    url = f"{config.OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": config.OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def call_mock(prompt: str, system: str = SYSTEM_PROMPT) -> str:
    """Mock LLM for offline testing — no API key needed."""
    context_start = prompt.find("CONTEXT FROM KNOWLEDGE BASE:")
    context_end = prompt.find("USER QUESTION:")
    excerpt = ""
    if context_start != -1 and context_end != -1:
        excerpt = prompt[context_start + 30:context_end].strip()[:200]

    return (
        "**[MOCK RESPONSE]**\n\n"
        "Context found in knowledge base:\n\n"
        f"{excerpt}...\n\n"
        "Set a real LLM_PROVIDER in .env to get real answers."
    )


# Provider registry — add new providers here only
_PROVIDER_MAP = {
    "gemini":    call_gemini,
    "anthropic": call_anthropic,
    "openai":    call_openai,
    "ollama":    call_ollama,
    "mock":      call_mock,
}


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE-PROVIDER ROUTER  (identical interface to original rag_pipeline.py)
# ─────────────────────────────────────────────────────────────────────────────

def call_llm(prompt: str, provider: Optional[str] = None) -> str:
    """
    Route to the appropriate LLM backend.

    Parameters
    ----------
    prompt   : full prompt string (system + context + question)
    provider : override config.LLM_PROVIDER for this call

    This function signature matches what rag_pipeline.py already calls,
    so RAGPipeline.run() needs zero changes.
    """
    p = (provider or config.LLM_PROVIDER).lower()

    if p not in _PROVIDER_MAP:
        raise ValueError(
            f"Unknown LLM provider: '{p}'. "
            f"Valid options: {', '.join(_PROVIDER_MAP.keys())}"
        )

    return _PROVIDER_MAP[p](prompt)


# ─────────────────────────────────────────────────────────────────────────────
# PARALLEL MULTI-LLM  (new — called by Streamlit UI for comparison view)
# ─────────────────────────────────────────────────────────────────────────────

def call_llm_parallel(
    prompt: str,
    providers: Optional[List[str]] = None,
    timeout: int = None,
) -> List[Dict[str, Any]]:
    """
    Run multiple LLMs simultaneously and return all answers.

    Uses ThreadPoolExecutor for true I/O concurrency — all API calls happen
    in parallel, so total time ≈ slowest single LLM instead of sum of all.

    Parameters
    ----------
    prompt    : full prompt string (same prompt sent to every LLM)
    providers : list of provider names, defaults to config.MULTI_LLM_PROVIDERS
    timeout   : seconds to wait per provider, defaults to config.MULTI_LLM_TIMEOUT

    Returns
    -------
    List of result dicts, one per provider:
    [
        {
            "provider":  "gemini",
            "answer":    "...",
            "success":   True,
            "error":     None,
            "elapsed_ms": 1240,
        },
        ...
    ]
    Results are returned in completion order (fastest LLM first).
    """
    providers = providers or config.MULTI_LLM_PROVIDERS
    timeout = timeout or config.MULTI_LLM_TIMEOUT
    results = []

    def _run_one(provider_name: str) -> Dict[str, Any]:
        t0 = time.time()
        try:
            answer = call_llm(prompt, provider=provider_name)
            return {
                "provider":   provider_name,
                "answer":     answer,
                "success":    True,
                "error":      None,
                "elapsed_ms": int((time.time() - t0) * 1000),
            }
        except Exception as exc:
            return {
                "provider":   provider_name,
                "answer":     "",
                "success":    False,
                "error":      str(exc),
                "elapsed_ms": int((time.time() - t0) * 1000),
            }

    # Submit all providers simultaneously
    with ThreadPoolExecutor(max_workers=len(providers)) as executor:
        future_to_provider = {
            executor.submit(_run_one, p): p
            for p in providers
        }

        for future in as_completed(future_to_provider, timeout=timeout):
            try:
                results.append(future.result())
            except TimeoutError:
                provider = future_to_provider[future]
                results.append({
                    "provider":   provider,
                    "answer":     "",
                    "success":    False,
                    "error":      f"Timed out after {timeout}s",
                    "elapsed_ms": timeout * 1000,
                })

    # Sort: successful answers first, then by speed
    results.sort(key=lambda r: (not r["success"], r["elapsed_ms"]))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_prompt = (
        "CONTEXT FROM KNOWLEDGE BASE:\n"
        "[1] SOURCE: Test Doc  (relevance: 0.95)\n"
        "Vector databases store embeddings for fast similarity search.\n\n"
        "USER QUESTION:\nWhat is a vector database?\n\nANSWER:"
    )

    print("Testing single-LLM call (mock):")
    answer = call_llm(test_prompt, provider="mock")
    print(answer[:200])

    print("\nTesting parallel call (mock × 2):")
    results = call_llm_parallel(test_prompt, providers=["mock", "mock"], timeout=10)
    for r in results:
        print(f"  {r['provider']}: {r['elapsed_ms']}ms | success={r['success']}")
