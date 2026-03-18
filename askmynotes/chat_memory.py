"""
chat_memory.py
─────────────────────────────────────────────────────────────────────────────
CHAT MEMORY MODULE — new in v2

WHY MEMORY?
───────────
Without memory, every question is answered in isolation:
  Turn 1: "What is RAG?"         → great answer
  Turn 2: "What are its benefits?" → "its" refers to nothing — bad retrieval

With memory:
  Turn 2 → query_rewriter rewrites "its benefits" → "benefits of RAG"
          → retrieval finds the right chunks
          → LLM answer references the prior discussion

HOW IT WORKS
────────────
  1. ChatMemory stores the last N turns (configurable via MEMORY_MAX_TURNS)
  2. It builds a "history string" injected into the LLM prompt so the model
     can reference prior exchanges without losing coherence
  3. It exposes the history list for query_rewriter.rewrite_standalone()
  4. Old turns are automatically dropped when the window is full

MEMORY IS NOT PASSED TO ENDEE
──────────────────────────────
Memory is a prompt engineering concern, not a retrieval concern. Endee always
does a fresh similarity search on the (possibly rewritten) query. We don't
store conversation history as vectors — that would pollute the knowledge base.
─────────────────────────────────────────────────────────────────────────────
"""

from datetime import datetime
from typing import List, Dict, Any, Optional

import config


class ChatMemory:
    """
    Sliding-window conversation history for the RAG chatbot.

    Keeps the last MEMORY_MAX_TURNS exchanges in memory.
    Provides formatted history strings for both the LLM prompt
    and the query rewriter.

    Usage
    -----
    memory = ChatMemory()
    memory.add_turn("user", "What is RAG?")
    memory.add_turn("assistant", "RAG stands for Retrieval-Augmented Generation...")

    history_str = memory.format_for_prompt()  # inject into LLM prompt
    history_list = memory.as_list()           # pass to query rewriter
    """

    def __init__(self, max_turns: int = None):
        self.max_turns = max_turns or config.MEMORY_MAX_TURNS
        self._turns: List[Dict[str, Any]] = []

    # ── Mutation ──────────────────────────────────────────────────────────────

    def add_turn(self, role: str, content: str, metadata: Dict = None):
        """
        Add one turn to the memory.

        Parameters
        ----------
        role     : "user" or "assistant"
        content  : message text
        metadata : optional dict (sources, timing, etc.) — stored but not
                   included in the prompt to keep context length manageable
        """
        self._turns.append({
            "role":      role,
            "content":   content,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "metadata":  metadata or {},
        })

        # Drop oldest turns when window is full
        # We keep max_turns *pairs* (user + assistant = 1 pair = 2 turns)
        max_raw = self.max_turns * 2
        if len(self._turns) > max_raw:
            self._turns = self._turns[-max_raw:]

    def clear(self):
        """Wipe all history."""
        self._turns.clear()

    # ── Read ──────────────────────────────────────────────────────────────────

    def as_list(self) -> List[Dict[str, str]]:
        """
        Return history as a plain list of {"role": ..., "content": ...} dicts.
        Used by query_rewriter.rewrite_standalone().
        """
        return [
            {"role": t["role"], "content": t["content"]}
            for t in self._turns
        ]

    def format_for_prompt(self, max_chars: int = 1200) -> str:
        """
        Format conversation history as a string to inject into the LLM prompt.

        Truncates old turns from the front if the total would exceed max_chars,
        ensuring the most recent context is always preserved.

        Returns empty string if there's no history (first question).
        """
        if not self._turns:
            return ""

        lines = []
        for turn in self._turns:
            role_label = "User" if turn["role"] == "user" else "Assistant"
            # Truncate very long assistant answers to save context space
            content = turn["content"]
            if turn["role"] == "assistant" and len(content) > 400:
                content = content[:400] + "…"
            lines.append(f"{role_label}: {content}")

        full = "\n".join(lines)

        # If over budget, keep only the tail (most recent turns)
        if len(full) > max_chars:
            full = "…[earlier turns omitted]\n" + full[-max_chars:]

        return full

    def build_prompt_with_memory(
        self,
        base_prompt: str,
        context: str,
        query: str,
    ) -> str:
        """
        Build a full LLM prompt that includes conversation history.

        Structure:
          CONTEXT FROM KNOWLEDGE BASE:
          [retrieved passages]

          CONVERSATION HISTORY:
          [last N turns]

          USER QUESTION:
          [current question]

          ANSWER:

        Parameters
        ----------
        base_prompt : not used (kept for signature compatibility)
        context     : formatted retrieved passages from build_context()
        query       : current user question (possibly rewritten)
        """
        history_str = self.format_for_prompt()

        parts = ["CONTEXT FROM KNOWLEDGE BASE:", "─" * 29, context, "─" * 29, ""]

        if history_str:
            parts += ["CONVERSATION HISTORY:", history_str, ""]

        parts += ["USER QUESTION:", query, "", "ANSWER:"]

        return "\n".join(parts)

    # ── Introspection ─────────────────────────────────────────────────────────

    @property
    def num_turns(self) -> int:
        return len(self._turns)

    @property
    def is_empty(self) -> bool:
        return len(self._turns) == 0

    def last_user_message(self) -> Optional[str]:
        """Return the most recent user message, or None."""
        for turn in reversed(self._turns):
            if turn["role"] == "user":
                return turn["content"]
        return None

    def last_assistant_message(self) -> Optional[str]:
        """Return the most recent assistant message, or None."""
        for turn in reversed(self._turns):
            if turn["role"] == "assistant":
                return turn["content"]
        return None

    def to_display_list(self) -> List[Dict[str, Any]]:
        """Return full turn list including timestamps for UI display."""
        return list(self._turns)
