#!/usr/bin/env python3
"""Ingest markdown documents → chunk → embed → store in Chroma.

Usage
-----
    python ingest.py [--data-dir data] [--persist-dir chroma_db] [--chunk-size 500] [--rebuild]
    python ingest.py --check                         # are sources and embedding stack current?
    python ingest.py --prune [--keep 2] [--dry-run]  # delete superseded collections

This reads all .md files under data-dir, splits them into chunks, embeds
each chunk, and stores them in a local Chroma collection. Files whose
content hash matches the active index reuse their stored embeddings, so only
added or changed files are embedded; --rebuild re-embeds everything, as does a
change of embedding model, sentence-transformers version, model revision or
chunk size.
"""

import argparse
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from uuid import uuid4

import chromadb
from chromadb.errors import NotFoundError

import policy

EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION_PREFIX = "observability-docs"
LEGACY_COLLECTION = COLLECTION_PREFIX  # indexes created before the active-collection pointer
_MANAGED_NAME = re.compile(rf"{re.escape(COLLECTION_PREFIX)}(-[0-9a-f]{{32}})?")


def embedding_versions() -> dict:
    """Identify the locally installed embedding stack without loading the model.

    ``embedding_revision`` is the Hugging Face snapshot (commit hash) of the
    model in the local cache, or None when it cannot be resolved offline, for
    example before the model is first downloaded.
    """
    try:
        st_version = version("sentence-transformers")
    except PackageNotFoundError:
        st_version = None
    try:
        from huggingface_hub import try_to_load_from_cache

        repo_id = EMBED_MODEL if "/" in EMBED_MODEL else f"sentence-transformers/{EMBED_MODEL}"
        cached = try_to_load_from_cache(repo_id, "config.json")
        revision = Path(cached).parent.name if isinstance(cached, str) else None
    except Exception:  # noqa: BLE001 - the revision is best-effort provenance
        revision = None
    return {"sentence_transformers_version": st_version, "embedding_revision": revision}


def embedding_mismatches(manifest: dict, installed: dict | None = None) -> list[str]:
    """Explain why stored embeddings may differ from what the installed stack
    would produce; an empty list means they are compatible.

    A manifest without a recorded sentence-transformers version (written before
    versions were recorded) counts as a mismatch, because its vectors cannot be
    shown to match the installed model. Revisions are compared only when both
    are known.
    """
    installed = installed or embedding_versions()
    reasons = []
    if manifest.get("embedding_model") != EMBED_MODEL:
        reasons.append(f"embedding model {manifest.get('embedding_model')!r} != {EMBED_MODEL!r}")
    recorded = manifest.get("sentence_transformers_version")
    if recorded is None:
        reasons.append("sentence-transformers version not recorded (index predates version tracking)")
    elif recorded != installed["sentence_transformers_version"]:
        reasons.append(f"sentence-transformers {recorded} != installed "
                       f"{installed['sentence_transformers_version']}")
    revision, current = manifest.get("embedding_revision"), installed["embedding_revision"]
    if revision and current and revision != current:
        reasons.append(f"model revision {revision} != cached {current}")
    return reasons


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


def _manifest(collection) -> dict | None:
    """Return a collection's corpus manifest, or None if absent or unsupported."""
    manifest = json.loads((collection.metadata or {}).get("corpus_manifest", "null"))
    if not manifest or manifest["schema_version"] != 1:
        return None
    return manifest


def _active_name(persist_dir: str) -> str:
    pointer = Path(persist_dir) / "active-collection"
    return pointer.read_text(encoding="utf-8") if pointer.exists() else LEGACY_COLLECTION


def _active_collection(persist_dir: str):
    """Return the active collection and its corpus manifest (None if unsupported)."""
    collection = chromadb.PersistentClient(path=persist_dir).get_collection(_active_name(persist_dir))
    return collection, _manifest(collection)


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
        installed = embedding_versions()
        mismatch = embedding_mismatches(manifest, installed)
        return {
            "status": "stale" if added or deleted or modified or mismatch else "current",
            "ingested_at": manifest["ingested_at"],
            "corpus_sha256": manifest["corpus_sha256"],
            "embedding_model": manifest["embedding_model"],
            "sentence_transformers_version": manifest.get("sentence_transformers_version"),
            "embedding_revision": manifest.get("embedding_revision"),
            "installed": installed,
            "embedding_mismatch": mismatch,
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

    # ── Policy front matter (issue #78): stripped from the chunked text ──
    records = {}
    for doc in docs:
        fields, doc["body"], errors = policy.parse_front_matter(doc["text"])
        records[doc["rel_path"]] = policy.validate(fields, errors)
    policies = policy.build_registry(records)
    for source, record in policies.items():
        for error in record["errors"]:
            print(f"Warning: {source}: {error} (treated as invalid policy metadata)")

    # ── Find an active index whose embeddings can be reused ──────
    versions = embedding_versions()
    previous, indexed = None, {}
    if not rebuild and Path(persist_dir).is_dir():
        reasons = ["no supported corpus manifest"]
        try:
            previous, previous_manifest = _active_collection(persist_dir)
            reasons = embedding_mismatches(previous_manifest, versions)
            if previous_manifest["chunk_size"] != chunk_size:
                reasons.append(f"chunk size {previous_manifest['chunk_size']} != {chunk_size}")
            if reasons:
                previous = None
            else:
                indexed = previous_manifest["sources"]
        except (OSError, UnicodeError, ValueError, KeyError, TypeError, AttributeError,
                NotFoundError):
            previous = None
        if previous is None:
            print(f"No compatible active index ({'; '.join(reasons)}); embedding all documents.")

    # ── Chunk, reusing embeddings of unchanged files ─────────────
    all_chunks: list[dict] = []
    embeddings: list[list[float] | None] = []
    added, changed = [], []
    for doc in docs:
        chunks = chunk_text_simple(doc["body"], chunk_size)
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
            and previous_manifest.get("policies") == policies
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
        versions = embedding_versions()  # the load may have downloaded the model
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
    name = f"{COLLECTION_PREFIX}-{uuid4().hex}"
    existing = {c.name for c in client.list_collections() if _MANAGED_NAME.fullmatch(c.name)}
    replaced = _active_name(persist_dir)
    if replaced not in existing:
        replaced = None
    manifest = {
        "schema_version": 1,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "data_dir": resolved_dir,
        "embedding_model": EMBED_MODEL,
        # Reused vectors keep their recorded revision when the cache cannot tell.
        "embedding_revision": versions["embedding_revision"] or (
            previous_manifest.get("embedding_revision") if previous is not None else None),
        "sentence_transformers_version": versions["sentence_transformers_version"],
        "chunk_size": chunk_size,
        # The collection this one replaces; --prune keeps it for rollback.
        "previous_collection": replaced,
        "sources": sources,
        # Validated policy front matter per source; applicability is decided
        # at query time (policy.applicability).
        "policies": policies,
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
    older = len(existing - {replaced})  # neither active nor kept for rollback by default
    if older:
        print(f"{older} older or incomplete collection(s) remain on disk; "
              "run `python ingest.py --prune` to delete them.")

    # Print a sample chunk for verification
    print("\nSample chunk:")
    print("─" * 50)
    print(texts[0][:500])
    print("─" * 50)
    return len(all_chunks)


def plan_prune(persist_dir: str, keep: int = 2) -> dict:
    """Decide which collections a prune keeps and removes, without deleting.

    The active collection is always kept. ``keep`` counts it, so the default 2
    also keeps the collection it replaced, for rollback; larger values follow
    each manifest's ``previous_collection`` further back. Where that chain ends
    early (history not recorded, a predecessor already pruned, or a rollback),
    the newest remaining collections by ingestion time fill the slots.
    Collections whose names this tool did not create are never touched.
    """
    if keep < 1:
        raise ValueError("--keep must be at least 1; the active collection is never pruned.")
    client = chromadb.PersistentClient(path=persist_dir)
    collections = {c.name: c for c in client.list_collections() if _MANAGED_NAME.fullmatch(c.name)}
    active = _active_name(persist_dir)
    if active not in collections:
        raise ValueError(f"Active collection {active!r} not found in {persist_dir}; nothing pruned.")

    def manifest_of(name):
        try:
            return _manifest(collections[name]) or {}
        except (ValueError, KeyError, TypeError):
            return {}

    kept, name = [active], active
    while len(kept) < keep:
        name = manifest_of(name).get("previous_collection")
        if name not in collections or name in kept:
            break
        kept.append(name)
    # History not recorded, already pruned, or rolled back past: rather than
    # delete more, fill remaining slots with the newest others (no manifest last).
    others = sorted((n for n in collections if n not in kept),
                    key=lambda n: manifest_of(n).get("ingested_at", ""), reverse=True)
    kept.extend(others[:keep - len(kept)])
    return {"active": active, "kept": kept,
            "remove": sorted(n for n in collections if n not in kept)}


_HNSW_FILES = {"header.bin", "data_level0.bin", "length.bin", "link_lists.bin",
               "index_metadata.pickle"}


def _orphaned_segment_dirs(persist_dir: str) -> list[Path]:
    """Vector-index directories no collection references any more.

    Chroma (1.5) deletes a collection's rows but leaves its HNSW directory on
    disk. A directory qualifies only if its name is a UUID absent from Chroma's
    segment table and it holds nothing but HNSW files; anything else is kept.
    """
    import sqlite3
    from uuid import UUID

    database = Path(persist_dir) / "chroma.sqlite3"
    try:
        with sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True) as db:
            referenced = {row[0] for row in db.execute("SELECT id FROM segments")}
    except sqlite3.Error:
        return []  # unknown layout: delete nothing
    orphans = []
    for path in Path(persist_dir).iterdir():
        try:
            UUID(path.name)
        except ValueError:
            continue
        if (path.is_dir() and not path.is_symlink() and path.name not in referenced
                and all(f.is_file() and f.name in _HNSW_FILES for f in path.iterdir())):
            orphans.append(path)
    return sorted(orphans)


def prune_collections(persist_dir: str, keep: int = 2, dry_run: bool = False) -> dict:
    """Delete superseded and incomplete collections as decided by plan_prune.

    Run only while no ingestion writes to the same index directory. A reader
    (CLI session or UI) still using a removed collection must reload.
    """
    if not Path(persist_dir).is_dir():
        raise ValueError(f"Index directory unavailable: {persist_dir}")
    plan = plan_prune(persist_dir, keep)
    removed = []
    if not dry_run:
        client = chromadb.PersistentClient(path=persist_dir)
        for name in plan["remove"]:
            if name == _active_name(persist_dir):  # re-read: never delete the active index
                continue
            client.delete_collection(name)
            removed.append(name)
    orphans = [] if dry_run else _orphaned_segment_dirs(persist_dir)
    freed = sum(f.stat().st_size for d in orphans for f in d.iterdir())
    for directory in orphans:
        shutil.rmtree(directory)
    return {"active": plan["active"], "kept": plan["kept"], "removed": removed,
            "would_remove": plan["remove"] if dry_run else [],
            "segment_dirs_removed": [d.name for d in orphans], "bytes_freed": freed,
            "dry_run": dry_run}


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
    parser.add_argument("--prune", action="store_true",
                        help="Delete superseded and incomplete collections (never the active one)")
    parser.add_argument("--keep", type=int, default=2,
                        help="With --prune: collections to keep, including the active one (default 2)")
    parser.add_argument("--dry-run", action="store_true",
                        help="With --prune: report what would be removed without deleting")
    args = parser.parse_args()
    if args.prune:
        try:
            result = prune_collections(args.persist_dir, args.keep, dry_run=args.dry_run)
        except ValueError as exc:
            print(exc)
            sys.exit(1)
        print(json.dumps(result, indent=2))
        return
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
