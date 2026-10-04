#!/usr/bin/env python3
"""Ingest markdown documents → chunk → embed → store in Chroma.

Usage
-----
    python ingest.py [--data-dir data] [--persist-dir chroma_db] [--chunk-size 500]

This reads all .md files under data-dir, splits them into chunks, embeds
each chunk, and stores them in a local Chroma collection.
"""

import argparse
import os
import re
import sys

from pathlib import Path

import chromadb


def load_documents(data_dir: str) -> list[dict]:
    """Read all .md files from data_dir and return a list of
    {'path': str, 'rel_path': str, 'filename': str, 'text': str} dicts."""
    docs = []
    root = Path(data_dir).resolve()
    for path in sorted(root.rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        if len(text.strip()) < 50:
            continue  # skip stubs / near-empty files
        rel = path.relative_to(root)
        docs.append({
            "path": str(path),
            "rel_path": str(rel),
            "filename": path.name,
            "text": text,
        })
    return docs


def chunk_text_simple(text: str, chunk_size: int) -> list[dict]:
    """Split a document into self-contained chunks with heading context.

    Groups paragraphs into chunks of up to chunk_size characters, where
    each chunk prepends its parent heading breadcrumb so it makes sense
    when retrieved in isolation (e.g. 'Coverage Targets > Tier 1').

    Paragraph boundaries are respected — a paragraph either fits entirely
    in the current chunk or starts a new one. No mid-paragraph splits.
    """
    lines = text.split("\n")
    heading_stack: list[str] = []
    chunks: list[dict] = []
    current_paragraphs: list[str] = []
    current_len = 0

    def _section_context() -> str:
        """Build a breadcrumb like 'Coverage Targets > Tier 1'."""
        return " > ".join(h for h in heading_stack if h) if heading_stack else ""

    def _flush() -> None:
        nonlocal current_paragraphs, current_len
        if not current_paragraphs:
            return
        ctx = _section_context()
        body = "\n\n".join(current_paragraphs)
        full = f"{ctx}\n\n{body}" if ctx else body
        chunks.append({
            "text": full.strip(),
            "section": heading_stack[-1] if heading_stack else "",
        })
        current_paragraphs = []
        current_len = 0

    for line in lines:
        heading_match = re.match(r"^(#{1,4})\s+(.+)", line)
        if heading_match:
            _flush()
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()
            target = level - 1  # # -> depth 0, ## -> depth 1, ### -> depth 2
            while len(heading_stack) > target:
                heading_stack.pop()
            heading_stack.append(title)
            continue

        stripped = line.strip()
        if not stripped:
            # blank line — paragraph boundary, keep going
            continue

        para_len = len(stripped)

        # If adding this paragraph would exceed chunk_size, flush first
        if current_len + para_len > chunk_size and current_paragraphs:
            _flush()

        current_paragraphs.append(stripped)
        current_len += para_len

    _flush()
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest docs into Chroma")
    parser.add_argument("--data-dir", default="data", help="Path to markdown files")
    parser.add_argument("--persist-dir", default="chroma_db",
                        help="Chroma persistence directory")
    parser.add_argument("--chunk-size", type=int, default=500,
                        help="Target chunk size in characters")
    args = parser.parse_args()

    # ── Load documents ───────────────────────────────────────────
    docs = load_documents(args.data_dir)
    if not docs:
        print(f"No markdown files found in {args.data_dir}/")
        sys.exit(1)
    print(f"Loaded {len(docs)} documents from {args.data_dir}/")

    # ── Chunk ────────────────────────────────────────────────────
    all_chunks: list[dict] = []
    for doc in docs:
        chunks = chunk_text_simple(doc["text"], args.chunk_size)
        for c in chunks:
            c["source"] = doc["rel_path"]
            c["source_file"] = doc["filename"]
            c["source_path"] = doc["path"]
        all_chunks.extend(chunks)
        print(f"  {doc['rel_path']}: {len(chunks)} chunks")

    print(f"\nTotal chunks: {len(all_chunks)}")

    # ── Embed and store ─────────────────────────────────────────
    from sentence_transformers import SentenceTransformer
    print("Loading embedding model...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    texts = [c["text"] for c in all_chunks]
    metadatas = [
        {
            "source": c["source"],
            "file": c["source_file"],
            "section": c.get("section", ""),
        }
        for c in all_chunks
    ]

    ids = [f"chunk-{i:04d}" for i in range(len(all_chunks))]

    print(f"Embedding {len(texts)} chunks...")
    embeddings = embedder.encode(texts, show_progress_bar=True).tolist()

    print(f"Storing in Chroma at {args.persist_dir}/")
    os.makedirs(args.persist_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=args.persist_dir)

    # Rebuild from scratch so re-running ingest never hits duplicate IDs
    try:
        client.delete_collection("observability-docs")
    except Exception:
        pass  # collection did not exist yet
    collection = client.create_collection(
        name="observability-docs",
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )

    print(f"Done — {len(all_chunks)} chunks in collection 'observability-docs'")

    # Print a sample chunk for verification
    print("\nSample chunk:")
    print("─" * 50)
    print(texts[0][:500])
    print("─" * 50)


if __name__ == "__main__":
    main()
