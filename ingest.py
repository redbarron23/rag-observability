#!/usr/bin/env python3
"""Ingest markdown documents → chunk → embed → store in Chroma.

Usage
-----
    python ingest.py [--data-dir data] [--persist-dir chroma_db] [--chunk-size 500] [--rebuild]

This reads all .md files under data-dir, splits them into chunks, embeds
each chunk, and stores them in a local Chroma collection. Files whose
content hash matches the active index reuse their stored embeddings, so only
added or changed files are embedded; --rebuild re-embeds everything.
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

import chromadb
from chromadb.errors import NotFoundError

EMBED_MODEL = "all-MiniLM-L6-v2"


def load_documents(data_dir: str) -> list[dict]:
    """Read all .md files from data_dir and return a list of
    {'path', 'rel_path', 'filename', 'text', 'sha256'} dicts."""
    docs = []
    root = Path(data_dir).resolve()

    def on_error(exc):
        raise exc  # An unreadable directory must not look like deleted documents.

    paths = []
    for directory, _, filenames in os.walk(root, onerror=on_error):
        paths.extend(Path(directory) / name for name in filenames if name.endswith(".md"))
    for path in sorted(paths):
        content = path.read_bytes()
        text = content.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
        if len(text.strip()) < 50:
            continue  # skip stubs / near-empty files
        rel = path.relative_to(root)
        docs.append({
            "path": str(path),
            "rel_path": str(rel),
            "filename": path.name,
            "text": text,
            "sha256": sha256(content).hexdigest(),
        })
    return docs


def _active_collection(persist_dir: str):
    """Return the active collection and its corpus manifest (None if unsupported)."""
    pointer = Path(persist_dir) / "active-collection"
    name = pointer.read_text(encoding="utf-8") if pointer.exists() else "observability-docs"
    collection = chromadb.PersistentClient(path=persist_dir).get_collection(name)
    manifest = json.loads((collection.metadata or {}).get("corpus_manifest", "null"))
    if not manifest or manifest["schema_version"] != 1:
        manifest = None
    return collection, manifest


def _reusable_embeddings(collection, source: str, source_sha256: str,
                         chunks: list[dict]) -> list[list[float]] | None:
    """Return stored embeddings for a document's chunks, or None unless every
    chunk is stored with identical text, section and source hash."""
    ids = [f"{source}#{i}" for i in range(len(chunks))]
    stored = collection.get(ids=ids, include=["embeddings", "documents", "metadatas"])
    by_id = dict(zip(stored["ids"], zip(stored["documents"], stored["metadatas"],
                                        stored["embeddings"], strict=False), strict=False))
    embeddings = []
    for chunk_id, chunk in zip(ids, chunks, strict=False):
        document, metadata, embedding = by_id.get(chunk_id, (None, None, None))
        metadata = metadata or {}
        if (document != chunk["text"] or metadata.get("source_sha256") != source_sha256
                or metadata.get("section") != chunk["section"]):
            return None
        embeddings.append([float(x) for x in embedding])
    return embeddings


def check_freshness(persist_dir: str, data_dir: str | None = None) -> dict:
    """Compare the active collection's source snapshot with local Markdown."""
    try:
        if not Path(persist_dir).is_dir():
            return {"status": "unavailable", "reason": f"Index directory unavailable: {persist_dir}"}
        _, manifest = _active_collection(persist_dir)
        if manifest is None:
            return {"status": "unavailable", "reason": "No supported corpus manifest; re-ingest."}
        root = Path(data_dir or manifest["data_dir"])
        if not root.is_dir():
            return {"status": "unavailable", "reason": f"Source directory unavailable: {root}"}
        current = {doc["rel_path"]: doc["sha256"] for doc in load_documents(str(root))}
        indexed = manifest["sources"]
        # Previously indexed files can become too short to qualify for ingestion.
        for source in indexed.keys() - current.keys():
            try:
                current[source] = sha256((root / source).read_bytes()).hexdigest()
            except FileNotFoundError:
                pass
        added = sorted(current.keys() - indexed.keys())
        deleted = sorted(indexed.keys() - current.keys())
        modified = sorted(source for source in current.keys() & indexed.keys()
                          if current[source] != indexed[source])
        return {
            "status": "stale" if added or deleted or modified else "current",
            "ingested_at": manifest["ingested_at"],
            "corpus_sha256": manifest["corpus_sha256"],
            "embedding_model": manifest["embedding_model"],
            "data_dir": str(root),
            "added": added, "modified": modified, "deleted": deleted,
        }
    except (OSError, UnicodeError, ValueError, KeyError, TypeError, AttributeError, NotFoundError) as exc:
        return {"status": "unavailable", "reason": str(exc)}


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
                     chunk_size: int = 500, rebuild: bool = False) -> int:
    """Index Markdown and return the number of stored chunks.

    Files whose hash matches the active index reuse its stored embeddings;
    added and changed files are embedded and deleted files are dropped.
    ``rebuild=True`` re-embeds every file. Any change is written to a new
    collection that is verified before the active-collection pointer switches,
    so a failure leaves the previous index active.
    """

    # ── Load documents ───────────────────────────────────────────
    docs = load_documents(data_dir)
    if not docs:
        raise ValueError(f"No markdown files found in {data_dir}/")
    print(f"Loaded {len(docs)} documents from {data_dir}/")
    resolved_dir = str(Path(data_dir).resolve())
    sources = {doc["rel_path"]: doc["sha256"] for doc in docs}

    # ── Find an active index whose embeddings can be reused ──────
    previous, indexed = None, {}
    if not rebuild and Path(persist_dir).is_dir():
        try:
            previous, previous_manifest = _active_collection(persist_dir)
            if (previous_manifest["embedding_model"] == EMBED_MODEL
                    and previous_manifest["chunk_size"] == chunk_size):
                indexed = previous_manifest["sources"]
            else:
                previous = None
        except (OSError, UnicodeError, ValueError, KeyError, TypeError, NotFoundError):
            previous = None
        if previous is None:
            print("No compatible active index; embedding all documents.")

    # ── Chunk, reusing embeddings of unchanged files ─────────────
    all_chunks: list[dict] = []
    embeddings: list[list[float] | None] = []
    added, changed = [], []
    for doc in docs:
        chunks = chunk_text_simple(doc["text"], chunk_size)
        for i, c in enumerate(chunks):
            c["index"] = i
            c["source"] = doc["rel_path"]
            c["source_file"] = doc["filename"]
            c["source_path"] = doc["path"]
            c["source_sha256"] = doc["sha256"]
        reused = None
        if previous is not None and chunks and indexed.get(doc["rel_path"]) == doc["sha256"]:
            reused = _reusable_embeddings(previous, doc["rel_path"], doc["sha256"], chunks)
        if reused is None:
            (changed if doc["rel_path"] in indexed else added).append(doc["rel_path"])
        all_chunks.extend(chunks)
        embeddings.extend(reused or [None] * len(chunks))
        status = "reused" if reused is not None else "embed"
        print(f"  {doc['rel_path']}: {len(chunks)} chunks ({status})")

    deleted = sorted(indexed.keys() - sources.keys())
    print(f"\nTotal chunks: {len(all_chunks)}")
    if previous is not None:
        print(f"Added: {len(added)}, changed: {len(changed)}, deleted: {len(deleted)}, "
              f"unchanged: {len(docs) - len(added) - len(changed)}")
    if not all_chunks:
        raise ValueError("No usable document text found; the index was not changed.")
    if (previous is not None and not (added or changed or deleted)
            and previous_manifest["data_dir"] == resolved_dir
            and previous.count() == len(all_chunks)):
        print("Index is up to date; nothing to embed.")
        return len(all_chunks)

    # ── Embed only new and changed chunks ────────────────────────
    to_embed = [i for i, embedding in enumerate(embeddings) if embedding is None]
    print(f"Reusing {len(all_chunks) - len(to_embed)} stored chunk embeddings.")
    if to_embed:
        from sentence_transformers import SentenceTransformer
        print("Loading embedding model...")
        embedder = SentenceTransformer(EMBED_MODEL)
        print(f"Embedding {len(to_embed)} chunks...")
        fresh = embedder.encode([all_chunks[i]["text"] for i in to_embed],
                                show_progress_bar=True).tolist()
        for i, embedding in zip(to_embed, fresh, strict=True):
            embeddings[i] = embedding

    texts = [c["text"] for c in all_chunks]
    metadatas = [
        {
            "source": c["source"],
            "file": c["source_file"],
            "section": c.get("section", ""),
            "source_sha256": c["source_sha256"],
        }
        for c in all_chunks
    ]

    # IDs are stable per document, so editing one file doesn't renumber
    # every chunk after it (e.g. "reference/coverage-targets.md#2")
    ids = [f"{c['source']}#{c['index']}" for c in all_chunks]

    print(f"Storing in Chroma at {persist_dir}/")
    os.makedirs(persist_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir)

    # Keep readers on the previous collection until the replacement is complete.
    name = f"observability-docs-{uuid4().hex}"
    manifest = {
        "schema_version": 1,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "data_dir": resolved_dir,
        "embedding_model": EMBED_MODEL,
        "embedding_revision": None,
        "chunk_size": chunk_size,
        "sources": sources,
        "corpus_sha256": sha256(json.dumps(sources, sort_keys=True).encode("utf-8")).hexdigest(),
    }
    collection = client.create_collection(
        name=name,
        metadata={"hnsw:space": "cosine", "corpus_manifest": json.dumps(manifest)},
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
    parser.add_argument("--data-dir", help="Path to Markdown (default: data; --check: recorded path)")
    parser.add_argument("--persist-dir", default="chroma_db",
                        help="Chroma persistence directory")
    parser.add_argument("--chunk-size", type=int, default=500,
                        help="Target chunk size in characters")
    parser.add_argument("--rebuild", action="store_true",
                        help="Re-embed every document instead of reusing unchanged embeddings")
    parser.add_argument("--check", action="store_true",
                        help="Check local source freshness without embedding or generation")
    args = parser.parse_args()
    if args.check:
        result = check_freshness(args.persist_dir, args.data_dir)
        print(json.dumps(result, indent=2))
        sys.exit({"current": 0, "stale": 1, "unavailable": 2}[result["status"]])
    try:
        ingest_documents(args.data_dir or "data", args.persist_dir, args.chunk_size,
                         rebuild=args.rebuild)
    except ValueError as exc:
        print(exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
