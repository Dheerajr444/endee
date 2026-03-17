"""
chatbot.py
─────────────────────────────────────────────────────────────────────────────
STEP 4 OF THE RAG PIPELINE – Terminal Chatbot

A rich terminal chatbot that:
  • Maintains multi-turn conversation history
  • Uses the full RAG pipeline for every response
  • Displays retrieved sources per answer
  • Supports special commands (/help, /history, /clear, /sources, /quit)

Run it:
  python chatbot.py

─────────────────────────────────────────────────────────────────────────────
"""

import sys
import time
from datetime import datetime
from typing import List, Dict, Any

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.table import Table
from rich.rule import Rule

import config
from rag_pipeline import RAGPipeline, display_rag_result

console = Console()


# ═════════════════════════════════════════════════════════════════════════════
# CONVERSATION HISTORY
# ═════════════════════════════════════════════════════════════════════════════

class ConversationHistory:
    """
    Simple in-memory conversation log.

    Stores each turn as a dict:
      {"role": "user"|"assistant", "content": str, "timestamp": str,
       "sources": list | None, "timing": dict | None}

    NOTE: This RAG implementation does NOT pass history back to the LLM.
    Each turn is answered independently from the knowledge base.
    For multi-turn reasoning, you would include prior exchanges in the LLM
    prompt — this is left as an extension exercise.
    """

    def __init__(self):
        self._turns: List[Dict[str, Any]] = []

    def add_user(self, text: str):
        self._turns.append({
            "role": "user",
            "content": text,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "sources": None,
            "timing": None,
        })

    def add_assistant(
        self,
        text: str,
        sources: List[Dict] = None,
        timing: Dict = None,
    ):
        self._turns.append({
            "role": "assistant",
            "content": text,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "sources": sources or [],
            "timing": timing or {},
        })

    def clear(self):
        self._turns.clear()

    @property
    def turns(self):
        return list(self._turns)

    @property
    def num_turns(self) -> int:
        return len(self._turns)


# ═════════════════════════════════════════════════════════════════════════════
# CHATBOT CLASS
# ═════════════════════════════════════════════════════════════════════════════

HELP_TEXT = """
[bold cyan]Available Commands[/bold cyan]

  [green]/help[/green]          Show this help message
  [green]/history[/green]       Print full conversation history
  [green]/sources[/green]       Show sources from the last response
  [green]/clear[/green]         Clear conversation history
  [green]/quit[/green]          Exit the chatbot
  [green]/index[/green]         Show information about the Endee index

  [dim]Any other input is treated as a question to the knowledge base.[/dim]
"""

class RAGChatbot:
    """
    Terminal-based conversational RAG chatbot.

    Features
    --------
    • Full RAG pipeline per query (embed → search Endee → LLM)
    • Conversation history stored in memory
    • Source citation display after each answer
    • Special slash commands for exploration and debugging
    """

    def __init__(self):
        console.print("[cyan]⚙  Initialising RAG pipeline…[/cyan]")
        self.pipeline = RAGPipeline()
        self.history = ConversationHistory()
        self._last_sources: List[Dict] = []
        console.print("[green]✓  Ready.[/green]\n")

    def _print_welcome(self):
        console.print(Panel.fit(
            "[bold cyan]🤖  RAG Document Chatbot[/bold cyan]\n\n"
            f"[dim]Knowledge base:  {config.ENDEE_INDEX_NAME}\n"
            f"Embedding model: {config.EMBEDDING_MODEL}\n"
            f"LLM provider:    {config.LLM_PROVIDER}\n"
            f"Top-K retrieval: {config.TOP_K_RESULTS} chunks[/dim]\n\n"
            "Ask me anything about the documents in the knowledge base.\n"
            "Type [bold green]/help[/bold green] for commands.",
            border_style="cyan",
            title="Endee RAG Chatbot",
        ))

    def _handle_command(self, command: str) -> bool:
        """
        Handle special /commands.

        Returns True if the input was a command (so the main loop can skip
        running the RAG pipeline), False otherwise.
        """
        cmd = command.lower().strip()

        if cmd == "/help":
            console.print(HELP_TEXT)
            return True

        elif cmd == "/clear":
            self.history.clear()
            console.print("[green]✓  Conversation history cleared.[/green]")
            return True

        elif cmd == "/history":
            if not self.history.turns:
                console.print("[dim]No conversation history yet.[/dim]")
                return True
            console.print(f"\n[bold]Conversation History ({self.history.num_turns} turns):[/bold]\n")
            for turn in self.history.turns:
                role_color = "cyan" if turn["role"] == "user" else "green"
                role_label = "You" if turn["role"] == "user" else "Bot"
                console.print(
                    f"[{role_color}][{turn['timestamp']}] {role_label}:[/{role_color}] "
                    f"{turn['content'][:120]}{'…' if len(turn['content']) > 120 else ''}"
                )
            console.print()
            return True

        elif cmd == "/sources":
            if not self._last_sources:
                console.print("[dim]No sources from the last response.[/dim]")
                return True
            console.print("\n[bold]Sources from last response:[/bold]")
            for i, src in enumerate(self._last_sources, 1):
                console.print(
                    f"\n[bold cyan][{i}] {src['title']}[/bold cyan]  "
                    f"[dim]similarity={src['similarity']:.3f}[/dim]\n"
                    f"    {src['text'][:200]}…"
                )
            console.print()
            return True

        elif cmd == "/index":
            info = self.pipeline.searcher.describe_index()
            table = Table(title=f"Endee Index: {config.ENDEE_INDEX_NAME}", show_header=False)
            table.add_column("Key", style="cyan")
            table.add_column("Value")
            for k, v in info.items():
                table.add_row(str(k), str(v))
            console.print(table)
            return True

        elif cmd in ("/quit", "/exit", "/q"):
            return None   # Signal to exit

        return False   # Not a command

    def chat(self, query: str) -> str:
        """
        Process one user query through the full RAG pipeline.

        Returns the assistant's answer string.
        """
        self.history.add_user(query)

        try:
            result = self.pipeline.run(query, top_k=config.TOP_K_RESULTS)
        except Exception as exc:
            error_msg = f"I encountered an error processing your question: {exc}"
            self.history.add_assistant(error_msg)
            return error_msg

        answer = result["answer"]
        sources = result.get("sources", [])
        timing = {
            "retrieval_ms": int(result["retrieval_time"] * 1000),
            "llm_ms": int(result["llm_time"] * 1000),
            "total_ms": int(result["total_time"] * 1000),
        }

        self.history.add_assistant(answer, sources=sources, timing=timing)
        self._last_sources = sources

        return answer, sources, timing

    def run(self):
        """Main interactive loop."""
        self._print_welcome()

        while True:
            try:
                user_input = console.input(
                    "\n[bold cyan]You >[/bold cyan] "
                ).strip()
            except (KeyboardInterrupt, EOFError):
                console.print("\n[dim]Use /quit to exit.[/dim]")
                continue

            if not user_input:
                continue

            # Handle commands
            if user_input.startswith("/"):
                result = self._handle_command(user_input)
                if result is None:   # /quit was entered
                    console.print(Panel.fit(
                        "[bold]Goodbye! 👋[/bold]\n"
                        f"[dim]Conversation had {self.history.num_turns} turns.[/dim]",
                        border_style="cyan",
                    ))
                    break
                continue

            # ── Run RAG pipeline ───────────────────────────────────────────────
            console.print("\n[dim]Searching knowledge base…[/dim]")

            try:
                answer, sources, timing = self.chat(user_input)
            except Exception as exc:
                console.print(f"[red]Error: {exc}[/red]")
                continue

            # ── Display answer ─────────────────────────────────────────────────
            console.print()
            console.print(Panel(
                Markdown(answer),
                title="[bold green]Assistant[/bold green]",
                border_style="green",
                padding=(1, 2),
            ))

            # ── Source summary ─────────────────────────────────────────────────
            if sources:
                source_names = [
                    f"[{i+1}] {s['title']} ({s['similarity']:.2f})"
                    for i, s in enumerate(sources[:3])
                ]
                console.print(
                    f"[dim]Sources: {' | '.join(source_names)}"
                    f"{'  +more' if len(sources) > 3 else ''}[/dim]"
                )
                console.print(
                    f"[dim]⏱  {timing['retrieval_ms']}ms retrieval + "
                    f"{timing['llm_ms']}ms LLM = {timing['total_ms']}ms total[/dim]"
                )
                console.print("[dim]Type /sources for full source passages.[/dim]")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    chatbot = RAGChatbot()
    chatbot.run()


if __name__ == "__main__":
    main()
