"""Local document explorer. Run with: streamlit run app.py"""

import os
from time import perf_counter

import streamlit as st
from chromadb.errors import NotFoundError

from ingest import ingest_documents, save_uploaded_documents
from query import (
    DEFAULT_PROVIDER,
    DEFAULT_RETRIEVAL,
    MAX_DISTANCE,
    N_RESULTS,
    PERSIST_DIR,
    PROVIDER_DEFAULT_MODELS,
    RETRIEVAL_MODES,
    RAGEngine,
    chunk_title,
)


@st.cache_resource
def load_engine(persist_dir: str, retrieval: str) -> RAGEngine:
    # Cached per (index directory, retrieval mode); Reload index clears every entry.
    return RAGEngine(persist_dir=persist_dir, retrieval=retrieval)


st.set_page_config(page_title="rag-observability", page_icon="🔎", layout="wide")
st.title("Observability document explorer")
st.caption("Ask a question, inspect the evidence, and optionally generate a cited answer.")

with st.sidebar:
    st.header("Retrieval settings")
    persist_dir = st.text_input("Index directory", value=PERSIST_DIR)
    retrieval = st.selectbox(
        "Retrieval mode", RETRIEVAL_MODES, index=RETRIEVAL_MODES.index(DEFAULT_RETRIEVAL),
        help="vector ranks by embedding similarity; hybrid also uses BM25 keyword matching, "
             "which helps with exact terms such as figures and product names.",
    )
    n_results = st.slider("Maximum chunks", min_value=1, max_value=12, value=N_RESULTS)
    max_distance = st.slider(
        "Maximum distance", min_value=0.0, max_value=2.0, value=MAX_DISTANCE, step=0.05,
        help="Lower distances mean closer matches. Set 2 to keep every retrieved chunk.",
    )
    if st.button("Reload index"):
        load_engine.clear()
        st.session_state.pop("result", None)
    provider = os.environ.get("LLM_PROVIDER", DEFAULT_PROVIDER)
    model = os.environ.get("LLM_MODEL") or PROVIDER_DEFAULT_MODELS.get(provider, "unknown")
    st.caption(f"Answer provider: {provider} · Model: {model}")
    with st.form("upload_documents"):
        st.subheader("Add documents")
        uploads = st.file_uploader(
            "Markdown files", type=["md"], accept_multiple_files=True, max_upload_size=5,
        )
        st.caption(
            "UTF-8 Markdown, at least 50 characters, up to 5 MB per file. "
            "Existing files are not overwritten. Saves locally and rebuilds the selected "
            "index from data/, including prior uploads."
        )
        upload_and_index = st.form_submit_button("Upload and index")

if upload_and_index:
    saved = False
    try:
        with st.spinner("Saving documents and rebuilding the local index…"):
            paths = save_uploaded_documents([(upload.name, upload.getvalue()) for upload in uploads])
            saved = True
            load_engine.clear()
            st.session_state.pop("result", None)
            count = ingest_documents(persist_dir=persist_dir)
        st.success(f"Saved {len(paths)} document(s) locally. Indexed {count} chunks; ready to search.")
    except Exception as exc:  # noqa: BLE001 — report validation, disk, and indexing failures in the UI
        if saved:
            st.error(f"Documents were saved, but indexing failed: {exc}. Retry Upload and index.")
        else:
            st.error(f"Upload failed: {exc}")

st.caption(
    "Find sources runs locally. Generate answer sends retrieved text to the configured "
    "provider and may incur charges. The bundled documents are fictional."
)
with st.form("question"):
    question = st.text_area(
        "Question", value="What is the coverage target for Tier 1 production resources?",
    )
    find_sources = st.form_submit_button("Find sources", type="primary")
    generate = st.form_submit_button("Generate answer")

if find_sources or generate:
    st.session_state.pop("result", None)
    if not question.strip():
        st.error("Enter a question first.")
    else:
        try:
            with st.spinner("Searching documents…" if find_sources else "Generating answer…"):
                engine = load_engine(persist_dir, retrieval)
                started = perf_counter()
                if generate:
                    answer, chunks, citations = engine.answer_with_citations(
                        question.strip(), n_results=n_results, max_distance=max_distance,
                    )
                else:
                    chunks = engine.retrieve(
                        question.strip(), n_results=n_results, max_distance=max_distance,
                    )
                    answer, citations = None, None
                st.session_state.result = {
                    "question": question.strip(), "answer": answer, "chunks": chunks,
                    "citations": citations, "retrieval": retrieval,
                    "seconds": perf_counter() - started,
                    "n_results": n_results, "max_distance": max_distance,
                }
        except NotFoundError:
            st.error("Index not found. Run python ingest.py for the selected index directory first.")
        except Exception as exc:  # noqa: BLE001 — keep query failures visible without crashing the app
            st.error(f"Query failed: {exc}")

if result := st.session_state.get("result"):
    st.subheader(result["question"])
    st.caption(
        f"{len(result['chunks'])} chunks · {result['seconds']:.2f}s "
        f"(excludes initial model loading) · {result['retrieval']} retrieval "
        f"· top {result['n_results']} "
        f"· distance ≤ {result['max_distance']:g}"
    )
    if result["answer"] is not None:
        st.markdown(result["answer"])
    if citations := result["citations"]:
        # Claude's native citations quote the exact supporting text; DeepSeek returns None.
        with st.expander(f"Quoted citations ({len(citations)})"):
            for number, citation in enumerate(citations, start=1):
                quote = " ".join(citation["cited_text"].split())
                st.markdown(f"**[{number}] {chunk_title(citation)}** · chunk {citation['chunk']}")
                st.text(quote)
    if not result["chunks"]:
        st.info("No relevant documents found. Try rephrasing or adjusting retrieval settings.")
    else:
        st.subheader("Retrieved evidence")
        for rank, chunk in enumerate(result["chunks"], start=1):
            distance = chunk["distance"]
            label = f"{rank}. {chunk['source']} — {chunk['section'] or 'Untitled section'}"
            if distance is not None:
                label += f" · distance {distance:.3f}"
            with st.expander(label, expanded=rank == 1):
                st.text(chunk["text"])
