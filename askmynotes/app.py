"""
app_v2.py  — Yupp-style parallel multi-LLM RAG UI
────────────────────────────────────────────────────────────────────────────
Every question fires ALL configured LLMs simultaneously.
Answers appear as equal-width cards in a single row, just like yupp.ai.

Layout
──────
┌─ Sidebar ──────────────────────────────────────────────────────────────────┐
│  ⚙ Settings · PDF upload · Doc list · Sources from last query             │
└────────────────────────────────────────────────────────────────────────────┘

┌─ Main area ─────────────────────────────────────────────────────────────────┐
│                                                                              │
│  ┌─ Turn 1 ──────────────────────────────────────────────────────────────┐  │
│  │  Q: What is a vector database?                                         │  │
│  │  ┌── Gemini ──┐  ┌── Claude ──┐  ┌── GPT-4o ──┐  ┌── Ollama ──┐     │  │
│  │  │ A vector…  │  │ A vector…  │  │ A vector…  │  │ A vector…  │     │  │
│  │  └────────────┘  └────────────┘  └────────────┘  └────────────┘     │  │
│  │  ⏱ retrieve 42ms  rerank 130ms  │ 📚 Sources                          │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌─ chat input ───────────────────────────────────────────────────────────┐ │
│  │  Ask a question…                                                    ➤  │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────┘

Launch:
  streamlit run app_v2.py
────────────────────────────────────────────────────────────────────────────
"""

import time
from typing import List, Dict, Any

import streamlit as st
import config

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AskMyNotes",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS  ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
html, body, [data-testid="stAppViewContainer"] { background: #0d0d14 !important; }
.block-container { padding: 1.2rem 1.5rem 4rem !important; max-width: 100% !important; }
section[data-testid="stSidebar"] > div { background: #111120 !important; }

/* ── Hide default streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Question bubble ── */
.q-block {
    background: #1a1f35;
    border-left: 3px solid #5577ff;
    border-radius: 10px;
    padding: 14px 18px;
    margin: 0 0 14px 0;
    font-size: 1.05em;
    color: #c8d4ff;
    font-weight: 500;
}

/* ── LLM answer card ── */
.llm-card {
    background: #13131f;
    border: 1px solid #222238;
    border-radius: 12px;
    padding: 14px 16px;
    height: 100%;
    display: flex;
    flex-direction: column;
    min-height: 180px;
}
.llm-card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 10px;
    padding-bottom: 8px;
    border-bottom: 1px solid #1e1e30;
}
.llm-name {
    font-weight: 700;
    font-size: 0.88em;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.llm-speed {
    font-size: 0.75em;
    color: #666688;
    background: #1a1a2e;
    border-radius: 8px;
    padding: 2px 8px;
}
.llm-answer {
    flex: 1;
    font-size: 0.9em;
    color: #c8c8e0;
    line-height: 1.65;
}
.llm-error {
    font-size: 0.85em;
    color: #ff6666;
    font-style: italic;
}

/* ── Color tags per LLM ── */
.tag-gemini   { color: #4db8ff; }
.tag-anthropic { color: #c084fc; }
.tag-openai   { color: #34d399; }
.tag-ollama   { color: #fb923c; }
.tag-mock     { color: #94a3b8; }

/* ── Metric bar ── */
.metric-row {
    display: flex; gap: 8px; flex-wrap: wrap;
    margin-top: 10px;
}
.m-chip {
    background: #16162a; border: 1px solid #2a2a45;
    border-radius: 20px; padding: 2px 10px;
    font-size: 0.72em; color: #7788aa;
}

/* ── Source card (sidebar) ── */
.src-card {
    background: #151522; border-left: 2px solid #334466;
    border-radius: 6px; padding: 8px 10px; margin: 5px 0;
    font-size: 0.78em; color: #8899bb;
}
.src-title { color: #aabbdd; font-weight: 600; }
.src-meta  { color: #556677; font-size: 0.85em; }

/* ── Rewrite banner ── */
.rewrite-banner {
    background: #11112a; border-left: 2px solid #6655cc;
    border-radius: 4px; padding: 5px 10px;
    font-size: 0.8em; color: #9988cc; margin-bottom: 8px;
}

/* ── PDF doc chip ── */
.doc-chip {
    display: inline-block;
    background: #181828; border: 1px solid #2a2a40;
    border-radius: 6px; padding: 4px 10px;
    font-size: 0.78em; color: #8899cc; margin: 2px 0;
    width: 100%;
}
.doc-chip-meta { color: #445566; font-size: 0.85em; }

/* ── Empty state ── */
.empty-state {
    text-align: center; padding: 80px 20px; color: #333355;
}
.empty-icon { font-size: 3.5em; margin-bottom: 12px; }
.empty-title { font-size: 1.4em; color: #445577; font-weight: 600; margin-bottom: 8px; }
.empty-sub { font-size: 0.9em; color: #334455; }

/* ── Score bar ── */
.sbar-bg { background:#1e1e30; border-radius:3px; height:4px; margin:4px 0; }
.sbar-fg { border-radius:3px; height:4px; }

/* ── Streamlit overrides ── */
.stChatInput textarea {
    background: #151525 !important;
    border: 1px solid #2a2a45 !important;
    color: #c8d0ff !important;
    border-radius: 12px !important;
}
div[data-testid="stChatInput"] > div {
    background: #0d0d18 !important;
    border-top: 1px solid #1a1a2e !important;
    padding: 10px 0 !important;
}
</style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═════════════════════════════════════════════════════════════════════════════

def _init():
    defaults = {
        "turns":           [],   # list of turn dicts
        "pdf_docs":        [],
        "total_queries":   0,
        "avg_retrieve_ms": 0.0,
        "chat_ready":      False,  # True once user clicks "Start asking questions"
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()


# ═════════════════════════════════════════════════════════════════════════════
# PIPELINE  (cached once)
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading RAG pipeline…")
def load_pipeline():
    from enhanced_rag_pipeline import EnhancedRAGPipeline
    try:
        p = EnhancedRAGPipeline(
            enable_rewriting=True,
            rewrite_strategy="standalone",
            enable_reranking=True,
            enable_memory=True,
        )
        return p, None
    except Exception as e:
        return None, str(e)


# ═════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════════

# Color per provider
LLM_COLORS = {
    "gemini":    "#4db8ff",
    "anthropic": "#c084fc",
    "openai":    "#34d399",
    "ollama":    "#fb923c",
    "mock":      "#94a3b8",
}

LLM_LABELS = {
    "gemini":    "Gemini",
    "anthropic": "Claude",
    "openai":    "GPT-4o",
    "ollama":    "Ollama",
    "mock":      "Mock",
}

def _color(provider: str) -> str:
    return LLM_COLORS.get(provider.lower(), "#aaaacc")

def _label(provider: str) -> str:
    return LLM_LABELS.get(provider.lower(), provider.upper())

def _chips(d: dict) -> str:
    bits = []
    if d.get("rewrite_time_ms") is not None:
        bits.append(f"✏️ rewrite {d['rewrite_time_ms']}ms")
    if d.get("retrieval_time_ms") is not None:
        bits.append(f"🔍 retrieve {d['retrieval_time_ms']}ms")
    if d.get("rerank_time_ms") is not None:
        bits.append(f"📊 rerank {d['rerank_time_ms']}ms")
    if d.get("total_time_ms") is not None:
        bits.append(f"⏱ total {d['total_time_ms']}ms")
    return "".join(f'<span class="m-chip">{b}</span>' for b in bits)


# ═════════════════════════════════════════════════════════════════════════════
# RENDER ONE TURN  (question + N answer cards)
# ═════════════════════════════════════════════════════════════════════════════

def render_turn(turn: Dict[str, Any]):
    """Render a single Q→{LLM answers} turn in yupp.ai card style."""

    # ── Question ──────────────────────────────────────────────────────────────
    st.markdown(
        f'<div class="q-block">💬 {turn["question"]}</div>',
        unsafe_allow_html=True,
    )

    # ── Query rewrite banner ──────────────────────────────────────────────────
    rq = turn.get("rewritten_query", "")
    oq = turn.get("question", "")
    if rq and rq.strip() != oq.strip():
        st.markdown(
            f'<div class="rewrite-banner">'
            f'✏️ <b>Rewritten for retrieval:</b> {rq}'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Answer cards ──────────────────────────────────────────────────────────
    llm_results = turn.get("llm_results", [])
    if llm_results:
        cols = st.columns(len(llm_results), gap="small")
        for col, r in zip(cols, llm_results):
            with col:
                provider  = r["provider"]
                color     = _color(provider)
                label     = _label(provider)
                speed     = f"{r['elapsed_ms']}ms"

                if r["success"]:
                    answer_html = r["answer"].replace("\n", "<br>")
                    body = f'<div class="llm-answer">{answer_html}</div>'
                else:
                    body = f'<div class="llm-error">⚠ {r["error"]}</div>'

                st.markdown(f"""
<div class="llm-card">
  <div class="llm-card-header">
    <span class="llm-name" style="color:{color}">{label}</span>
    <span class="llm-speed">{speed}</span>
  </div>
  {body}
</div>
""", unsafe_allow_html=True)

    # ── Metrics + sources toggle ───────────────────────────────────────────────
    st.markdown(
        f'<div class="metric-row">{_chips(turn)}</div>',
        unsafe_allow_html=True,
    )

    sources = turn.get("sources", [])
    if sources:
        with st.expander(f"📚 {len(sources)} sources retrieved", expanded=False):
            for i, s in enumerate(sources, 1):
                sim   = s.get("similarity", 0)
                rrs   = s.get("rerank_score")
                page  = s.get("page", "")
                fname = s.get("filename", "")
                title = s.get("title", "")
                color = "#4caf50" if sim > 0.7 else ("#ff9800" if sim > 0.4 else "#f44336")
                rr_str = f" · rerank {rrs:.2f}" if rrs is not None else ""
                page_str = f" · p.{page}" if page else ""
                fname_str = f" · {fname}" if fname else ""

                st.markdown(f"""
<div class="src-card">
  <div class="src-title">[{i}] {title}</div>
  <div class="src-meta">{fname_str}{page_str}{rr_str}
    <span style="color:{color}"> ● {sim:.3f}</span>
  </div>
  <div class="sbar-bg">
    <div class="sbar-fg" style="width:{int(sim*100)}%;background:{color}"></div>
  </div>
  <div style="margin-top:5px;color:#667799">
    {s.get("text","")[:220]}{"…" if len(s.get("text","")) > 220 else ""}
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown(
        '<hr style="border:none;border-top:1px solid #1a1a28;margin:16px 0">',
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════

def render_sidebar(pipeline):
    with st.sidebar:
        st.markdown("## 🧠 AskMyNotes")
        st.caption("Upload notes. Ask questions. Get AI answers.")
        st.divider()

        # ── Active LLMs ───────────────────────────────────────────────────────
        st.markdown("### 🤖 Active LLMs")
        all_providers = ["gemini", "anthropic", "openai", "ollama", "mock"]
        defaults = config.MULTI_LLM_PROVIDERS

        selected = []
        for p in all_providers:
            color = _color(p)
            label = _label(p)
            checked = st.checkbox(
                f"**{label}**",
                value=(p in defaults),
                key=f"llm_{p}",
            )
            if checked:
                selected.append(p)

        if not selected:
            st.warning("Select at least one LLM.")
        st.session_state["active_providers"] = selected

        # ── Pipeline settings ─────────────────────────────────────────────────
        st.divider()
        st.markdown("### ⚙️ Retrieval Settings")

        top_k = st.slider("Top-K passages", 1, 10, config.TOP_K_RESULTS)
        st.session_state["top_k"] = top_k

        rw_on = st.toggle("Query rewriting (Gemini)", value=True)
        rk_on = st.toggle("Cross-encoder reranking",  value=True)

        if rw_on:
            strat = st.selectbox(
                "Rewrite strategy",
                ["standalone", "expand", "hyde", "combined"],
                help=(
                    "standalone = use chat history to clarify follow-ups\n"
                    "expand = add related keywords\n"
                    "hyde = generate hypothetical answer as query\n"
                    "combined = standalone then expand"
                ),
            )
            st.session_state["rewrite_strategy"] = strat
        else:
            st.session_state["rewrite_strategy"] = "none"

        if pipeline:
            pipeline.enable_rewriting = rw_on
            pipeline.enable_reranking = rk_on
            pipeline.rewrite_strategy = st.session_state.get("rewrite_strategy", "standalone")

        # ── Stats ─────────────────────────────────────────────────────────────
        st.divider()
        st.markdown("### 📊 Session Stats")
        c1, c2 = st.columns(2)
        c1.metric("Questions", st.session_state["total_queries"])
        c2.metric(
            "Avg retrieve",
            f"{st.session_state['avg_retrieve_ms']:.0f}ms"
            if st.session_state["total_queries"] else "—"
        )
        if pipeline:
            st.metric("Memory turns", pipeline.memory.num_turns)

        # ── Controls ──────────────────────────────────────────────────────────
        st.divider()
        if st.button("🗑️ Clear chat + memory", use_container_width=True):
            st.session_state["turns"] = []
            st.session_state["total_queries"] = 0
            if pipeline:
                pipeline.clear_memory()
            st.rerun()

        # ── PDF upload ────────────────────────────────────────────────────────
        st.divider()
        st.markdown("### 📄 Knowledge Base")
        st.caption("Upload PDFs — they are indexed into Endee immediately.")

        uploaded = st.file_uploader(
            "Choose PDFs", type=["pdf"],
            accept_multiple_files=True, key="uploader",
        )
        if uploaded:
            for f in uploaded:
                done = [d["filename"] for d in st.session_state["pdf_docs"]]
                if f.name in done:
                    continue
                with st.spinner(f"Indexing {f.name}…"):
                    try:
                        from pdf_loader import ingest_pdf
                        bar = st.progress(0, text="Starting…")
                        def _p(step, pct): bar.progress(pct, text=step)
                        res = ingest_pdf(f.getvalue(), filename=f.name, progress_callback=_p)
                        bar.empty()
                        if res["success"]:
                            st.session_state["pdf_docs"].append({
                                "filename":    res["filename"],
                                "pages":       res["pages"],
                                "chunks":      res["chunks"],
                                "elapsed_sec": res["elapsed_sec"],
                            })
                            st.success(f"✅ {res['filename']} — {res['chunks']} chunks")
                        else:
                            st.error(f"❌ {res['error']}")
                    except Exception as e:
                        st.error(f"❌ {e}")

        # ── Doc list ──────────────────────────────────────────────────────────
        if st.session_state["pdf_docs"]:
            for i, doc in enumerate(st.session_state["pdf_docs"]):
                c1, c2 = st.columns([5, 1])
                c1.markdown(
                    f'<div class="doc-chip">📄 <b>{doc["filename"]}</b><br>'
                    f'<span class="doc-chip-meta">'
                    f'{doc["pages"]}p · {doc["chunks"]} chunks</span></div>',
                    unsafe_allow_html=True,
                )
                if c2.button("✕", key=f"rm_{i}"):
                    st.session_state["pdf_docs"].pop(i)
                    st.rerun()

        # ── Example questions ─────────────────────────────────────────────────
        


# ═════════════════════════════════════════════════════════════════════════════
# CENTERED PDF UPLOAD SCREEN  (shown when no PDFs uploaded yet)
# ═════════════════════════════════════════════════════════════════════════════

def render_upload_screen():
    """
    Full-page centered upload screen — shown before any PDF is uploaded.
    The chat input and all LLM cards are hidden until at least one PDF
    is successfully indexed into Endee.
    """
    # Extra CSS only for this screen
    st.markdown("""
<style>
/* Hide chat input on upload screen */
div[data-testid="stChatInput"] { display: none !important; }

.upload-wrapper {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 72vh;
    text-align: center;
    padding: 20px;
}
.upload-icon  { font-size: 4em; margin-bottom: 16px; }
.upload-title {
    font-size: 1.9em; font-weight: 700; color: #c8d4ff;
    margin-bottom: 8px;
}
.upload-sub {
    font-size: 0.95em; color: #445566;
    margin-bottom: 32px; max-width: 460px; line-height: 1.6;
}
.upload-box {
    background: #111120;
    border: 2px dashed #2a2a50;
    border-radius: 16px;
    padding: 40px 48px;
    width: 100%;
    max-width: 520px;
    transition: border-color .2s;
}
.upload-box:hover { border-color: #5577ff; }
.upload-note {
    margin-top: 20px; font-size: 0.78em; color: #334455;
}
.llm-pills {
    display: flex; gap: 8px; justify-content: center;
    flex-wrap: wrap; margin-top: 24px;
}
.llm-pill {
    border-radius: 20px; padding: 4px 14px;
    font-size: 0.78em; font-weight: 600;
    border: 1px solid;
}
</style>
""", unsafe_allow_html=True)

    # Centered wrapper — use columns to center in Streamlit
    _, center, _ = st.columns([1, 2, 1])

    with center:
        st.markdown("""
<div class="upload-wrapper">
  <div class="upload-icon">🧠</div>
  <div class="upload-title">AskMyNotes</div>
  <div class="upload-sub">
    Upload notes. Ask questions. Get AI answers.<br><br>
    Drop your PDF notes below — they'll be indexed instantly.<br>
    Then all AIs answer your questions simultaneously.
  </div>
</div>
""", unsafe_allow_html=True)

        # ── File uploader inside the box ──────────────────────────────────────
        st.markdown('<div class="upload-box">', unsafe_allow_html=True)

        uploaded_files = st.file_uploader(
            "Drop PDF files here or click to browse",
            type=["pdf"],
            accept_multiple_files=True,
            key="center_uploader",
            label_visibility="visible",
        )

        st.markdown('</div>', unsafe_allow_html=True)

        # ── Process uploads ───────────────────────────────────────────────────
        if uploaded_files:
            for f in uploaded_files:
                done = [d["filename"] for d in st.session_state["pdf_docs"]]
                if f.name in done:
                    st.success(f"✅ {f.name} already indexed")
                    continue

                bar = st.progress(0, text=f"Indexing {f.name}…")

                def _prog(step, pct):
                    bar.progress(pct, text=step)

                try:
                    from pdf_loader import ingest_pdf
                    res = ingest_pdf(
                        f.getvalue(),
                        filename=f.name,
                        progress_callback=_prog,
                    )
                    bar.empty()

                    if res["success"]:
                        st.session_state["pdf_docs"].append({
                            "filename":    res["filename"],
                            "pages":       res["pages"],
                            "chunks":      res["chunks"],
                            "elapsed_sec": res["elapsed_sec"],
                        })
                        st.success(
                            f"✅ **{res['filename']}** indexed — "
                            f"{res['pages']} pages · {res['chunks']} chunks · "
                            f"{res['elapsed_sec']}s"
                        )
                    else:
                        bar.empty()
                        st.error(f"❌ {res['filename']}: {res['error']}")

                except ImportError:
                    bar.empty()
                    st.error("pdf_loader.py not found in project folder.")
                except Exception as e:
                    bar.empty()
                    st.error(f"❌ {e}")

        # ── Once at least one PDF is indexed, show "Start chatting" button ────
        if st.session_state["pdf_docs"]:
            st.markdown("<br>", unsafe_allow_html=True)
            # Show indexed docs summary
            for doc in st.session_state["pdf_docs"]:
                st.markdown(
                    f'📄 **{doc["filename"]}** — '
                    f'{doc["pages"]} pages · {doc["chunks"]} chunks',
                )
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button(
                "⚡ Start asking questions →",
                use_container_width=True,
                type="primary",
            ):
                st.session_state["chat_ready"] = True
                st.rerun()

        # ── LLM pills at bottom ───────────────────────────────────────────────
        active = st.session_state.get("active_providers", config.MULTI_LLM_PROVIDERS)
        pills_html = '<div class="llm-pills">'
        for p in active:
            c = _color(p)
            pills_html += (
                f'<span class="llm-pill" style="color:{c};border-color:{c}22;'
                f'background:{c}11">{_label(p)}</span>'
            )
        pills_html += '</div>'
        st.markdown(pills_html, unsafe_allow_html=True)

        st.markdown(
            '<p style="text-align:center;font-size:0.75em;color:#223344;'
            'margin-top:16px">Powered by Endee vector database</p>',
            unsafe_allow_html=True,
        )


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    pipeline, err = load_pipeline()
    render_sidebar(pipeline)

    if err:
        st.error(f"Pipeline failed to load: {err}")
        st.info("Run `docker compose up -d` then `python embed_documents.py`")
        return

    # ── GATE: show upload screen until at least one PDF is ready ─────────────
    has_docs   = len(st.session_state["pdf_docs"]) > 0
    chat_ready = st.session_state.get("chat_ready", False)

    if not has_docs or not chat_ready:
        render_upload_screen()
        return   # ← chat input and arena are completely hidden

    # ══════════════════════════════════════════════════════════════════════════
    # CHAT ARENA  (only reached after PDF uploaded + button clicked)
    # ══════════════════════════════════════════════════════════════════════════

    # ── Header ────────────────────────────────────────────────────────────────
    active = st.session_state.get("active_providers", config.MULTI_LLM_PROVIDERS)
    labels = " · ".join(_label(p) for p in active)
    st.markdown("## 🧠 AskMyNotes")
    st.caption(
        f"Upload notes. Ask questions. Get AI answers. — "
        f"Firing: **{labels}** · {len(st.session_state['pdf_docs'])} note(s) loaded"
    )
    st.divider()

    # ── Render all previous turns ─────────────────────────────────────────────
    for turn in st.session_state["turns"]:
        render_turn(turn)

    # ── Idle state (no turns yet, but PDFs loaded) ────────────────────────────
    if not st.session_state["turns"]:
        docs_list = " · ".join(
            f"📄 {d['filename']}" for d in st.session_state["pdf_docs"]
        )
        st.markdown(f"""
<div class="empty-state">
  <div class="empty-icon">⚡</div>
  <div class="empty-title">Knowledge base ready</div>
  <div class="empty-sub">
    {docs_list}<br><br>
    Ask any question below — all LLMs answer simultaneously.
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Chat input ────────────────────────────────────────────────────────────
    inject = st.session_state.pop("inject_query", None)
    prompt = st.chat_input("Ask anything about your documents…") or inject

    if not prompt:
        return

    providers = st.session_state.get("active_providers", config.MULTI_LLM_PROVIDERS)
    if not providers:
        st.warning("Select at least one LLM in the sidebar.")
        return

    top_k    = st.session_state.get("top_k", config.TOP_K_RESULTS)
    strategy = st.session_state.get("rewrite_strategy", "standalone")
    pipeline.rewrite_strategy = strategy

    # ── Show question immediately ─────────────────────────────────────────────
    st.markdown(
        f'<div class="q-block">💬 {prompt}</div>',
        unsafe_allow_html=True,
    )

    # ── Skeleton loading cards ────────────────────────────────────────────────
    cols = st.columns(len(providers), gap="small")
    placeholders = []
    for col, p in zip(cols, providers):
        with col:
            ph = st.empty()
            ph.markdown(f"""
<div class="llm-card">
  <div class="llm-card-header">
    <span class="llm-name" style="color:{_color(p)}">{_label(p)}</span>
    <span class="llm-speed">…</span>
  </div>
  <div class="llm-answer" style="color:#333355;font-style:italic">Thinking…</div>
</div>
""", unsafe_allow_html=True)
            placeholders.append(ph)

    # ── Fire all LLMs in parallel ─────────────────────────────────────────────
    with st.spinner(""):
        try:
            result = pipeline.run_parallel(prompt, top_k=top_k, providers=providers)
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            return

    # Clear skeleton cards
    for ph in placeholders:
        ph.empty()

    # ── Render real answer cards ──────────────────────────────────────────────
    llm_results = result.get("llm_results", [])
    cols = st.columns(len(llm_results), gap="small")
    for col, r in zip(cols, llm_results):
        with col:
            color = _color(r["provider"])
            label = _label(r["provider"])
            speed = f"{r['elapsed_ms']}ms"
            if r["success"]:
                answer_html = r["answer"].replace("\n", "<br>")
                body = f'<div class="llm-answer">{answer_html}</div>'
            else:
                body = f'<div class="llm-error">⚠ {r["error"]}</div>'

            st.markdown(f"""
<div class="llm-card">
  <div class="llm-card-header">
    <span class="llm-name" style="color:{color}">{label}</span>
    <span class="llm-speed">{speed}</span>
  </div>
  {body}
</div>
""", unsafe_allow_html=True)

    # ── Rewrite banner ────────────────────────────────────────────────────────
    rq = result.get("rewritten_query", "")
    oq = result.get("original_query", prompt)
    if rq and rq.strip() != oq.strip():
        st.markdown(
            f'<div class="rewrite-banner">✏️ <b>Rewritten:</b> {rq}</div>',
            unsafe_allow_html=True,
        )

    # ── Metrics row ───────────────────────────────────────────────────────────
    st.markdown(
        f'<div class="metric-row">{_chips(result)}</div>',
        unsafe_allow_html=True,
    )

    # ── Sources expander ──────────────────────────────────────────────────────
    sources = result.get("sources", [])
    if sources:
        with st.expander(f"📚 {len(sources)} sources", expanded=False):
            for i, s in enumerate(sources, 1):
                sim    = s.get("similarity", 0)
                rrs    = s.get("rerank_score")
                page   = s.get("page", "")
                fname  = s.get("filename", "")
                title  = s.get("title", "")
                color  = "#4caf50" if sim > 0.7 else ("#ff9800" if sim > 0.4 else "#f44336")
                rr_str    = f" · rerank {rrs:.2f}" if rrs is not None else ""
                page_str  = f" · p.{page}" if page else ""
                fname_str = f" · {fname}" if fname else ""
                st.markdown(f"""
<div class="src-card">
  <div class="src-title">[{i}] {title}</div>
  <div class="src-meta">{fname_str}{page_str}{rr_str}
    <span style="color:{color}"> ● {sim:.3f}</span>
  </div>
  <div class="sbar-bg">
    <div class="sbar-fg" style="width:{int(sim*100)}%;background:{color}"></div>
  </div>
  <div style="margin-top:4px;color:#667799">
    {s.get("text","")[:200]}{"…" if len(s.get("text","")) > 200 else ""}
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown(
        '<hr style="border:none;border-top:1px solid #1a1a28;margin:16px 0">',
        unsafe_allow_html=True,
    )

    # ── Save turn ─────────────────────────────────────────────────────────────
    st.session_state["turns"].append({
        "question":          prompt,
        "rewritten_query":   rq,
        "original_query":    oq,
        "llm_results":       llm_results,
        "sources":           sources,
        "rewrite_time_ms":   result.get("rewrite_time_ms"),
        "retrieval_time_ms": result.get("retrieval_time_ms"),
        "rerank_time_ms":    result.get("rerank_time_ms"),
        "total_time_ms":     result.get("total_time_ms"),
    })

    # ── Update stats ──────────────────────────────────────────────────────────
    n    = st.session_state["total_queries"] + 1
    prev = st.session_state["avg_retrieve_ms"]
    r_ms = result.get("retrieval_time_ms", 0) or 0
    st.session_state["total_queries"]   = n
    st.session_state["avg_retrieve_ms"] = (prev * (n - 1) + r_ms) / n


if __name__ == "__main__":
    main()
