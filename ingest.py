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
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

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
    heading_stack: list[tuple[int, str]] = []
    chunks: list[dict] = []
    paragraph_lines: list[str] = []
    current_paragraphs: list[str] = []
    current_len = 0

    def _section_context() -> str:
        """Build a breadcrumb like 'Coverage Targets > Tier 1'."""
        return " > ".join(title for _, title in heading_stack)

    def _flush() -> None:
        nonlocal current_paragraphs, current_len
        if not current_paragraphs:
            return
        ctx = _section_context()
        body = "\n\n".join(current_paragraphs)
        full = f"{ctx}\n\n{body}" if ctx else body
        chunks.append({
            "text": full.strip(),
            "section": heading_stack[-1][1] if heading_stack else "",
        })
        current_paragraphs = []
        current_len = 0

    def _add_paragraph() -> None:
        nonlocal current_len
        if not paragraph_lines:
            return
        paragraph = "\n".join(paragraph_lines).strip()
        separator_len = 2 if current_paragraphs else 0
        if current_paragraphs and current_len + separator_len + len(paragraph) > chunk_size:
            _flush()
            separator_len = 0
        current_paragraphs.append(paragraph)
        current_len += separator_len + len(paragraph)
        paragraph_lines.clear()

    for line in lines:
        heading_match = re.match(r"^(#{1,4})\s+(.+)", line)
        if heading_match:
            _add_paragraph()
            _flush()
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            continue

        if line.strip():
            paragraph_lines.append(line)
        else:
            _add_paragraph()

    _add_paragraph()
    _flush()
    return chunks


def save_uploaded_documents(files: list[tuple[str, bytes]], data_dir: str = "data") -> list[Path]:
    """Validate the whole batch, then save Markdown without replacing existing files."""
    if not files:
        raise ValueError("Choose at least one Markdown file.")
    validated = []
    for name, content in files:
        if (Path(name).name != name or "/" in name or "\\" in name
                or "\x00" in name or len(name.encode("utf-8")) > 255
                or Path(name).suffix.lower() != ".md"):
            raise ValueError("Each upload must have a plain filename ending in .md.")
        if len(content) > 5 * 1024 * 1024:
            raise ValueError(f"{name}: maximum file size is 5 MB.")
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{name}: save the document as UTF-8 text.") from exc
        if "\x00" in text or len(text.strip()) < 50 or not chunk_text_simple(text, 500):
            raise ValueError(f"{name}: include at least 50 characters of usable Markdown text.")
        path = Path(data_dir) / "uploads" / sha256(content).hexdigest() / (Path(name).stem + ".md")
        validated.append((path, text.encode("utf-8")))
    for path, content in validated:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("xb") as target:
                target.write(content)
        except FileExistsError:
            if path.read_bytes() != content:
                raise ValueError(f"{path.name}: an existing upload has different contents.") from None
    return [path for path, _ in validated]


def ingest_documents(data_dir: str = "data", persist_dir: str = "chroma_db",
                     chunk_size: int = 500) -> int:
    """Rebuild the index from Markdown and return the number of stored chunks."""

    # ── Load documents ───────────────────────────────────────────
    docs = load_documents(data_dir)
    if not docs:
        raise ValueError(f"No markdown files found in {data_dir}/")
    print(f"Loaded {len(docs)} documents from {data_dir}/")

    # ── Chunk ────────────────────────────────────────────────────
    all_chunks: list[dict] = []
    for doc in docs:
        chunks = chunk_text_simple(doc["text"], chunk_size)
        for i, c in enumerate(chunks):
            c["index"] = i
            c["source"] = doc["rel_path"]
            c["source_file"] = doc["filename"]
            c["source_path"] = doc["path"]
        all_chunks.extend(chunks)
        print(f"  {doc['rel_path']}: {len(chunks)} chunks")

    print(f"\nTotal chunks: {len(all_chunks)}")
    if not all_chunks:
        raise ValueError("No usable document text found; the index was not changed.")

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

    # IDs are stable per document, so editing one file doesn't renumber
    # every chunk after it (e.g. "reference/coverage-targets.md#2")
    ids = [f"{c['source']}#{c['index']}" for c in all_chunks]

    print(f"Embedding {len(texts)} chunks...")
    embeddings = embedder.encode(texts, show_progress_bar=True).tolist()

    print(f"Storing in Chroma at {persist_dir}/")
    os.makedirs(persist_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir)

    # Keep readers on the previous collection until the replacement is complete.
    name = f"observability-docs-{uuid4().hex}"
    collection = client.create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    if collection.count() != len(ids):
        raise RuntimeError("Incomplete replacement index; the active index was not changed.")

    pointer = Path(persist_dir) / "active-collection"
    pending = pointer.with_name(f".{name}.tmp")
    try:
        pending.write_text(name, encoding="utf-8")
        os.replace(pending, pointer)
    finally:
        pending.unlink(missing_ok=True)

    print(f"Done — {len(all_chunks)} chunks in collection '{name}'")

    # Print a sample chunk for verification
    print("\nSample chunk:")
    print("─" * 50)
    print(texts[0][:500])
    print("─" * 50)
    return len(all_chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest docs into Chroma")
    parser.add_argument("--data-dir", default="data", help="Path to markdown files")
    parser.add_argument("--persist-dir", default="chroma_db",
                        help="Chroma persistence directory")
    parser.add_argument("--chunk-size", type=int, default=500,
                        help="Target chunk size in characters")
    args = parser.parse_args()
    try:
        ingest_documents(args.data_dir, args.persist_dir, args.chunk_size)
    except ValueError as exc:
        print(exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
