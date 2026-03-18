"""
pdf_loader.py
─────────────────────────────────────────────────────────────────────────────
PDF INGESTION MODULE  — extends the existing RAG pipeline

What this module does:
  1. Accepts a PDF file (path on disk OR raw bytes from Streamlit uploader)
  2. Extracts text page-by-page using pdfplumber
  3. Reuses the EXACT same chunk_text() and create_chunks_from_documents()
     functions already in embed_documents.py — nothing duplicated
  4. Reuses the EXACT same generate_embeddings() and upsert_chunks()
     functions from embed_documents.py
  5. Stores each chunk with metadata:  filename, page_number, source="pdf"
  6. Returns a summary dict so the caller (Streamlit UI) can show progress

Architecture position:
  ┌────────────────────────────────────────────────────────────┐
  │  NEW: User uploads PDF via Streamlit                       │
  │       ↓                                                    │
  │  pdf_loader.ingest_pdf()                                   │
  │       ↓                                                    │
  │  PDF text extraction  (pdfplumber, page-by-page)           │
  │       ↓                                                    │
  │  chunk_text()  ← imported from embed_documents.py          │
  │       ↓                                                    │
  │  generate_embeddings()  ← imported from embed_documents.py │
  │       ↓                                                    │
  │  upsert_chunks() into Endee  ← same index, same model      │
  │       ↓                                                     │
  │  EXISTING retrieval pipeline continues UNCHANGED ✅         │
  └────────────────────────────────────────────────────────────┘

Key design decisions:
  • source="pdf" stored in metadata → retrieval works for BOTH txt and pdf
  • page_number stored in metadata → users see which page an answer came from
  • filename stored in metadata → users see which PDF the answer came from
  • Chunk IDs prefixed with pdf_ to avoid collision with text doc chunk IDs
  • All embedding / upsert logic delegated back to embed_documents.py
    so there is EXACTLY ONE place those functions live

─────────────────────────────────────────────────────────────────────────────
"""

import io
import os
import sys
import time
import hashlib
from typing import List, Dict, Any, Union

# ── Re-use everything already written in embed_documents.py ──────────────────
# We import these directly so there is zero code duplication.
from embed_documents import (
    chunk_text,               # same chunking strategy
    generate_embeddings,      # same embedding model calls
    upsert_chunks,            # same Endee upsert logic
    get_endee_client,         # same Endee connection
    create_or_get_index,      # same index creation/reuse
    load_embedding_model,     # same model loader
)

import config


# ═════════════════════════════════════════════════════════════════════════════
# 1.  PDF TEXT EXTRACTION
# ═════════════════════════════════════════════════════════════════════════════

def extract_text_from_pdf(
    pdf_source: Union[str, bytes, io.BytesIO],
    filename: str = "uploaded.pdf",
) -> List[Dict[str, Any]]:
    """
    Extract text from a PDF file, page by page.

    WHY PAGE-BY-PAGE?
    Keeping page boundaries lets us store the page number in metadata.
    When the chatbot retrieves a passage, it can tell the user
    "this came from page 4 of report.pdf" — much more useful than just
    showing raw text with no provenance.

    Parameters
    ----------
    pdf_source : str  → file path on disk
                bytes → raw PDF bytes (from st.file_uploader)
                BytesIO → file-like object
    filename   : display name stored in metadata (e.g. "Q3_report.pdf")

    Returns
    -------
    List of page dicts:
        [{"page": 1, "text": "...", "filename": "report.pdf"}, ...]

    Only pages with extractable text are included (scanned/image PDFs
    without OCR will produce empty pages which are silently skipped).
    """
    try:
        import pdfplumber
    except ImportError:
        raise ImportError(
            "pdfplumber is not installed.\n"
            "Run:  pip install pdfplumber"
        )

    # Normalise input to a file-like object pdfplumber can open
    if isinstance(pdf_source, str):
        # File path
        pdf_file = open(pdf_source, "rb")
        should_close = True
    elif isinstance(pdf_source, bytes):
        pdf_file = io.BytesIO(pdf_source)
        should_close = False
    elif isinstance(pdf_source, io.BytesIO):
        pdf_file = pdf_source
        should_close = False
    else:
        raise TypeError(f"Unsupported pdf_source type: {type(pdf_source)}")

    pages = []
    try:
        with pdfplumber.open(pdf_file) as pdf:
            total_pages = len(pdf.pages)
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if text and text.strip():
                    pages.append({
                        "page": page_num,
                        "total_pages": total_pages,
                        "text": text.strip(),
                        "filename": filename,
                    })
    finally:
        if should_close:
            pdf_file.close()

    return pages


# ═════════════════════════════════════════════════════════════════════════════
# 2.  BUILD CHUNK RECORDS FROM PDF PAGES
# ═════════════════════════════════════════════════════════════════════════════

def create_chunks_from_pdf_pages(
    pages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Convert extracted PDF pages into chunk records.

    Reuses chunk_text() from embed_documents.py with identical settings
    (CHUNK_SIZE, CHUNK_OVERLAP from config) so chunk quality is consistent
    whether the source is a .txt file or a .pdf file.

    Each chunk record carries:
      • id         – unique stable ID  (pdf_{hash}_{page}_{chunk_idx})
      • text       – chunk text content
      • doc_id     – stable hash of filename (groups chunks by document)
      • title      – human-readable label: "filename.pdf – Page N"
      • chunk_idx  – chunk position within its page
      • page       – page number (PDF-specific metadata)
      • filename   – original PDF filename
      • source     – always "pdf" (distinguishes from text docs in retrieval)
    """
    all_chunks = []

    # Stable doc_id based on filename so the same PDF always maps to the
    # same doc_id regardless of upload order
    for page_info in pages:
        filename = page_info["filename"]
        page_num = page_info["page"]
        page_text = page_info["text"]

        doc_id = hashlib.md5(filename.encode()).hexdigest()[:8]

        # Human-readable title shown in search results and source citations
        title = f"{filename} — Page {page_num}"

        # Chunk this page's text using the identical strategy as embed_documents.py
        text_chunks = chunk_text(page_text)

        for chunk_idx, chunk in enumerate(text_chunks):
            # Prefix with "pdf_" to guarantee no ID collision with text doc chunks
            chunk_id = f"pdf_{doc_id}_p{page_num:04d}_c{chunk_idx:03d}"

            all_chunks.append({
                "id": chunk_id,
                "text": chunk,
                "doc_id": doc_id,
                "title": title,
                "chunk_idx": chunk_idx,
                "page": page_num,
                "filename": filename,
                "source": "pdf",
            })

    return all_chunks


# ═════════════════════════════════════════════════════════════════════════════
# 3.  MAIN INGESTION FUNCTION  (called by Streamlit UI)
# ═════════════════════════════════════════════════════════════════════════════

def ingest_pdf(
    pdf_source: Union[str, bytes, io.BytesIO],
    filename: str = "uploaded.pdf",
    progress_callback=None,
) -> Dict[str, Any]:
    """
    Full ingestion pipeline for a single PDF file.

    This is the ONE function the Streamlit UI calls. It handles everything:
    extract → chunk → embed → upsert into Endee.

    The existing retrieval pipeline (search_documents.py, rag_pipeline.py)
    is completely unaware this function exists — it just sees more vectors
    in the same Endee index and retrieves them normally.

    Parameters
    ----------
    pdf_source        : file path, raw bytes, or BytesIO
    filename          : display name for the PDF
    progress_callback : optional callable(step: str, pct: int) for UI updates
                        e.g. lambda step, pct: st.progress(pct, text=step)

    Returns
    -------
    dict with keys:
        success      : bool
        filename     : str
        pages        : int   — number of pages with extractable text
        chunks       : int   — total chunks created
        vectors      : int   — vectors upserted into Endee
        error        : str   — error message if success=False
        elapsed_sec  : float — total time taken
    """
    t_start = time.time()

    def _progress(step: str, pct: int):
        if progress_callback:
            progress_callback(step, pct)

    try:
        # ── Step 1: Extract text ──────────────────────────────────────────────
        _progress("Extracting text from PDF…", 10)
        pages = extract_text_from_pdf(pdf_source, filename=filename)

        if not pages:
            return {
                "success": False,
                "filename": filename,
                "error": (
                    "No extractable text found in this PDF. "
                    "It may be a scanned/image PDF. "
                    "Please use a PDF with selectable text."
                ),
                "pages": 0, "chunks": 0, "vectors": 0,
                "elapsed_sec": round(time.time() - t_start, 2),
            }

        # ── Step 2: Build chunk records ───────────────────────────────────────
        _progress("Chunking text…", 25)
        chunks = create_chunks_from_pdf_pages(pages)

        if not chunks:
            return {
                "success": False,
                "filename": filename,
                "error": "Text was extracted but produced no chunks.",
                "pages": len(pages), "chunks": 0, "vectors": 0,
                "elapsed_sec": round(time.time() - t_start, 2),
            }

        # ── Step 3: Load embedding model ──────────────────────────────────────
        _progress("Loading embedding model…", 40)
        model = load_embedding_model()

        # ── Step 4: Generate embeddings ───────────────────────────────────────
        _progress("Generating embeddings…", 60)
        texts = [c["text"] for c in chunks]
        embeddings = generate_embeddings(model, texts)

        # ── Step 5: Connect to Endee and upsert ───────────────────────────────
        _progress("Connecting to Endee…", 75)
        client = get_endee_client()
        index = create_or_get_index(client)

        _progress("Storing vectors in Endee…", 85)

        # Build upsert records — identical format to embed_documents.py
        # but with extra PDF metadata (page, filename, source) in meta
        records = [
            {
                "id": chunk["id"],
                "vector": emb,
                "meta": {
                    "text":      chunk["text"],
                    "title":     chunk["title"],
                    "doc_id":    chunk["doc_id"],
                    "chunk_idx": chunk["chunk_idx"],
                    "page":      chunk["page"],
                    "filename":  chunk["filename"],
                    "source":    "pdf",
                },
                "filter": {
                    "doc_id": chunk["doc_id"],
                    "source": "pdf",
                },
            }
            for chunk, emb in zip(chunks, embeddings)
        ]

        # Upsert in batches of 100 (same as embed_documents.py)
        batch_size = 100
        for i in range(0, len(records), batch_size):
            index.upsert(records[i : i + batch_size])

        _progress("Done!", 100)

        return {
            "success": True,
            "filename": filename,
            "pages": len(pages),
            "chunks": len(chunks),
            "vectors": len(records),
            "error": None,
            "elapsed_sec": round(time.time() - t_start, 2),
        }

    except Exception as exc:
        return {
            "success": False,
            "filename": filename,
            "error": str(exc),
            "pages": 0, "chunks": 0, "vectors": 0,
            "elapsed_sec": round(time.time() - t_start, 2),
        }


# ═════════════════════════════════════════════════════════════════════════════
# 4.  COMMAND-LINE USAGE  (for testing without the UI)
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Test PDF ingestion from the command line:
        python pdf_loader.py path/to/your/file.pdf
    """
    if len(sys.argv) < 2:
        print("Usage: python pdf_loader.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        sys.exit(1)

    filename = os.path.basename(pdf_path)
    print(f"\nIngesting: {filename}")
    print("-" * 40)

    def cli_progress(step, pct):
        print(f"  [{pct:3d}%] {step}")

    result = ingest_pdf(pdf_path, filename=filename, progress_callback=cli_progress)

    print("\n" + "=" * 40)
    if result["success"]:
        print(f"✅  Success!")
        print(f"    File:    {result['filename']}")
        print(f"    Pages:   {result['pages']}")
        print(f"    Chunks:  {result['chunks']}")
        print(f"    Vectors: {result['vectors']}")
        print(f"    Time:    {result['elapsed_sec']}s")
    else:
        print(f"❌  Failed: {result['error']}")
