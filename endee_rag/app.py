"""
app.py
─────────────────────────────────────────────────────────────────────────────
STREAMLIT WEB INTERFACE – RAG Document Chatbot

Launch with:
  streamlit run app.py

Features:
  • Chat interface with message history
  • Sidebar showing retrieved sources per answer
  • Expandable source passages with similarity scores
  • Visual similarity score bars
  • Performance timing display
  • Configuration panel in sidebar

─────────────────────────────────────────────────────────────────────────────
"""

import time
import sys
from typing import List, Dict, Any

import streamlit as st

import config

# ── Page config – must be first Streamlit call ────────────────────────────────
st.set_page_config(
    page_title="RAG Chatbot – Endee",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ═════════════════════════════════════════════════════════════════════════════
# CSS – polished dark theme
# ═════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
/* ── Main layout ── */
.stApp { background: #0f1117; }
.block-container { padding-top: 2rem; max-width: 900px; }

/* ── Chat messages ── */
.user-msg {
    background: #1e3a5f;
    border-left: 3px solid #4a9eff;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
    color: #e8f4ff;
}
.bot-msg {
    background: #1a2a1a;
    border-left: 3px solid #4caf50;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
    color: #e8ffe8;
}

/* ── Source cards ── */
.source-card {
    background: #1e1e2e;
    border: 1px solid #333;
    border-radius: 6px;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 0.85em;
}

/* ── Score bar ── */
.score-bar-bg {
    background: #2a2a3e;
    border-radius: 4px;
    height: 6px;
    width: 100%;
    margin-top: 4px;
}
.score-bar-fill {
    background: linear-gradient(90deg, #4caf50, #8bc34a);
    border-radius: 4px;
    height: 6px;
}

/* ── Timing badge ── */
.timing-badge {
    display: inline-block;
    background: #252540;
    border: 1px solid #444;
    border-radius: 12px;
    padding: 2px 10px;
    font-size: 0.75em;
    color: #9999cc;
    margin: 2px 3px;
}

/* ── Chat input ── */
.stChatInput textarea {
    background: #1e1e2e !important;
    color: #e0e0ff !important;
    border: 1px solid #444 !important;
}

/* ── Sidebar ── */
.css-1d391kg { background: #0d0d1a !important; }

/* ── Metric ── */
.stMetric { background: #1e1e2e; border-radius: 8px; padding: 8px; }
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALISATION
# ═════════════════════════════════════════════════════════════════════════════

def init_session():
    defaults = {
        "messages": [],          # List of {role, content, sources, timing}
        "pipeline": None,        # RAGPipeline instance (cached across reruns)
        "pipeline_ready": False,
        "total_queries": 0,
        "avg_retrieval_ms": 0.0,
        "avg_llm_ms": 0.0,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session()


# ═════════════════════════════════════════════════════════════════════════════
# CACHED PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading RAG pipeline…")
def load_pipeline():
    """
    Load and cache the RAGPipeline.
    @st.cache_resource means this runs only ONCE across all user sessions.
    The heavy parts (model download, index connection) happen here.
    """
    from rag_pipeline import RAGPipeline
    try:
        pipeline = RAGPipeline()
        return pipeline, None
    except SystemExit as e:
        return None, f"Pipeline initialisation failed. Is the Endee server running?"
    except Exception as e:
        return None, str(e)


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        st.markdown("## 🔍 RAG Chatbot")
        st.markdown("*Powered by Endee Vector DB*")
        st.divider()

        # ── Configuration ─────────────────────────────────────────────────────
        st.markdown("### ⚙️ Configuration")
        st.markdown(f"""
| Setting | Value |
|---------|-------|
| Index | `{config.ENDEE_INDEX_NAME}` |
| Model | `{config.EMBEDDING_MODEL}` |
| Dimensions | `{config.EMBEDDING_DIMENSION}` |
| LLM | `{config.LLM_PROVIDER}` |
| Top-K | `{config.TOP_K_RESULTS}` |
""")

        # ── Top-K slider ──────────────────────────────────────────────────────
        st.divider()
        top_k = st.slider(
            "Top-K Results",
            min_value=1, max_value=10,
            value=config.TOP_K_RESULTS,
            help="Number of document chunks retrieved per query",
        )
        st.session_state["top_k_override"] = top_k

        # ── Session stats ─────────────────────────────────────────────────────
        st.divider()
        st.markdown("### 📊 Session Stats")
        n = st.session_state["total_queries"]
        col1, col2 = st.columns(2)
        col1.metric("Queries", n)
        col2.metric(
            "Avg Retrieval",
            f"{st.session_state['avg_retrieval_ms']:.0f}ms" if n else "—"
        )

        # ── Clear button ──────────────────────────────────────────────────────
        st.divider()
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state["messages"] = []
            st.session_state["total_queries"] = 0
            st.rerun()

        # ── Example queries ───────────────────────────────────────────────────
        st.divider()
        st.markdown("### 💡 Example Questions")
        examples = [
            "What is a vector database?",
            "How does RAG work?",
            "Explain cosine similarity",
            "What is all-MiniLM-L6-v2?",
            "How do I run Endee with Docker?",
            "What is the difference between AI and ML?",
        ]
        for q in examples:
            if st.button(q, use_container_width=True, key=f"ex_{q[:20]}"):
                st.session_state["example_query"] = q
                st.rerun()

        # ── Links ─────────────────────────────────────────────────────────────
        st.divider()
        st.markdown("### 🔗 Resources")
        st.markdown("""
- [Endee Docs](https://docs.endee.io)
- [Endee GitHub](https://github.com/endee-io/endee)
- [sentence-transformers](https://sbert.net)
- [Anthropic Claude](https://console.anthropic.com)
""")


# ═════════════════════════════════════════════════════════════════════════════
# SOURCE PANEL
# ═════════════════════════════════════════════════════════════════════════════

def render_sources(sources: List[Dict[str, Any]]):
    """Render retrieved sources in an expander."""
    if not sources:
        return

    with st.expander(f"📚 Retrieved Sources ({len(sources)})", expanded=False):
        for i, src in enumerate(sources, 1):
            score = src["similarity"]
            score_pct = int(score * 100)
            score_color = "#4caf50" if score > 0.7 else ("#ff9800" if score > 0.4 else "#f44336")

            st.markdown(f"""
<div class="source-card">
  <strong>[{i}] {src['title']}</strong>
  &nbsp;&nbsp;<span style="color:{score_color}; font-size:0.85em;">
    similarity: {score:.3f}
  </span>
  <div class="score-bar-bg">
    <div class="score-bar-fill" style="width:{score_pct}%; background: {score_color};"></div>
  </div>
  <p style="margin:6px 0 0 0; color:#aaa; font-size:0.82em;">
    {src['text'][:250]}{'…' if len(src['text']) > 250 else ''}
  </p>
</div>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# MESSAGE DISPLAY
# ═════════════════════════════════════════════════════════════════════════════

def render_messages():
    """Render all messages in the chat history."""
    for msg in st.session_state["messages"]:
        if msg["role"] == "user":
            with st.chat_message("user", avatar="🧑"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.write(msg["content"])
                # Timing badges
                if msg.get("timing"):
                    t = msg["timing"]
                    st.markdown(
                        f'<span class="timing-badge">🔍 {t["retrieval_ms"]}ms</span>'
                        f'<span class="timing-badge">🧠 {t["llm_ms"]}ms</span>'
                        f'<span class="timing-badge">⏱ {t["total_ms"]}ms total</span>',
                        unsafe_allow_html=True,
                    )
                render_sources(msg.get("sources", []))


# ═════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═════════════════════════════════════════════════════════════════════════════

def main():
    render_sidebar()

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("## 🤖 RAG Document Chatbot")
    st.markdown(
        f"*Ask questions about the knowledge base – "
        f"answers grounded in documents via **Endee** vector search.*"
    )
    st.divider()

    # ── Load pipeline ──────────────────────────────────────────────────────────
    pipeline, error = load_pipeline()

    if error:
        st.error(f"⚠️ Pipeline Error: {error}")
        st.info("""
**Troubleshooting:**
1. Make sure Docker is running: `docker ps`
2. Start Endee: `docker compose up -d`
3. Embed your documents: `python embed_documents.py`
4. Reload this page
        """)
        return

    # ── Handle example query injected from sidebar ─────────────────────────────
    prefill = st.session_state.pop("example_query", None)

    # ── Render chat history ────────────────────────────────────────────────────
    render_messages()

    # ── Chat input ─────────────────────────────────────────────────────────────
    prompt = st.chat_input("Ask a question about the knowledge base…") or prefill

    if prompt:
        top_k = st.session_state.get("top_k_override", config.TOP_K_RESULTS)

        # Append user message
        st.session_state["messages"].append({
            "role": "user",
            "content": prompt,
            "sources": [],
            "timing": {},
        })

        with st.chat_message("user", avatar="🧑"):
            st.write(prompt)

        # ── Run RAG pipeline ──────────────────────────────────────────────────
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Searching knowledge base…"):
                try:
                    result = pipeline.run(prompt, top_k=top_k)
                    answer = result["answer"]
                    sources = result.get("sources", [])
                    timing = {
                        "retrieval_ms": int(result["retrieval_time"] * 1000),
                        "llm_ms": int(result["llm_time"] * 1000),
                        "total_ms": int(result["total_time"] * 1000),
                    }
                except Exception as exc:
                    answer = f"⚠️ Error: {exc}"
                    sources = []
                    timing = {}

            st.write(answer)

            if timing:
                st.markdown(
                    f'<span class="timing-badge">🔍 {timing["retrieval_ms"]}ms</span>'
                    f'<span class="timing-badge">🧠 {timing["llm_ms"]}ms</span>'
                    f'<span class="timing-badge">⏱ {timing["total_ms"]}ms total</span>',
                    unsafe_allow_html=True,
                )
            render_sources(sources)

        # ── Persist to session state ───────────────────────────────────────────
        st.session_state["messages"].append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "timing": timing,
        })

        # ── Update stats ──────────────────────────────────────────────────────
        n = st.session_state["total_queries"] + 1
        st.session_state["total_queries"] = n
        prev_avg = st.session_state["avg_retrieval_ms"]
        st.session_state["avg_retrieval_ms"] = (
            (prev_avg * (n - 1) + timing.get("retrieval_ms", 0)) / n
        )

    # ── Empty state ────────────────────────────────────────────────────────────
    if not st.session_state["messages"]:
        st.markdown("""
<div style="text-align:center; padding:40px; color:#666;">
  <div style="font-size:3em; margin-bottom:16px;">🔍</div>
  <h3 style="color:#888;">Ask anything about the knowledge base</h3>
  <p>Try one of the example questions in the sidebar, or type your own.</p>
  <p style="font-size:0.85em; color:#555;">
    Documents are embedded in Endee. Answers are grounded in retrieved passages.
  </p>
</div>
""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
