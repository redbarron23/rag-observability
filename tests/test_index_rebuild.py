"""Index replacement safety and the provenance of each indexed snapshot."""

import json
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from unittest.mock import Mock

import chromadb
import numpy as np
import pytest
from chromadb.api.models.Collection import Collection

import ingest
from query import RAGEngine


@pytest.fixture
def index(tmp_path, monkeypatch):
    embedder = Mock()
    embedder.encode.return_value.tolist.return_value = [[1.0, 0.0]]
    monkeypatch.setattr("sentence_transformers.SentenceTransformer", Mock(return_value=embedder))
    monkeypatch.setattr("query.SentenceTransformer", Mock(return_value=embedder))
    data = tmp_path / "data"
    data.mkdir()
    (data / "old.md").write_text("# Old\n\n" + "Original runbook content. " * 3)
    persist = tmp_path / "index"
    ingest.ingest_documents(str(data), str(persist))
    return data, persist, RAGEngine(str(persist))


@pytest.mark.parametrize("failure", ["write", "validate", "switch"])
def test_failed_rebuild_keeps_previous_queries(index, monkeypatch, failure):
    data, persist, reader = index
    expected = reader.retrieve("runbook")
    active = (persist / "active-collection").read_text()
    (data / "old.md").unlink()
    (data / "new.md").write_text("# New\n\n" + "Replacement instructions. " * 3)
    if failure == "write":
        original_add = Collection.add

        def partial_write(self, **kwargs):
            original_add(self, **kwargs)
            raise RuntimeError("injected write failure")

        monkeypatch.setattr(Collection, "add", partial_write)
    elif failure == "validate":
        monkeypatch.setattr(Collection, "count", lambda self: 0)
    else:
        monkeypatch.setattr(ingest.os, "replace", Mock(side_effect=OSError("disk full")))
    with pytest.raises((RuntimeError, OSError)):
        ingest.ingest_documents(str(data), str(persist))
    assert (persist / "active-collection").read_text() == active
    assert reader.retrieve("runbook") == expected
    assert RAGEngine(str(persist)).retrieve("runbook") == expected
    assert json.loads(reader.collection.metadata["corpus_manifest"])["corpus_sha256"] == (
        ingest.check_freshness(str(persist))["corpus_sha256"]
    )
    assert not list(persist.glob(".*.tmp"))


def test_successful_rebuild_removes_deleted_sources_and_keeps_existing_reader(index):
    data, persist, reader = index
    (data / "old.md").unlink()
    (data / "new.md").write_text("# New\n\n" + "Replacement instructions. " * 3)
    ingest.ingest_documents(str(data), str(persist))
    assert {c["source"] for c in RAGEngine(str(persist)).retrieve("runbook")} == {"new.md"}
    assert {c["source"] for c in reader.retrieve("runbook")} == {"old.md"}


def test_legacy_index_remains_readable(tmp_path, monkeypatch):
    monkeypatch.setattr("query.SentenceTransformer", Mock())
    client = chromadb.PersistentClient(path=str(tmp_path))
    client.create_collection("observability-docs")
    assert RAGEngine(str(tmp_path)).collection.name == "observability-docs"
    assert ingest.check_freshness(str(tmp_path))["status"] == "unavailable"


def test_manifest_matches_indexed_bytes_and_is_stable_across_rebuilds(index):
    data, persist, reader = index
    manifest = json.loads(reader.collection.metadata["corpus_manifest"])
    assert manifest["sources"] == {"old.md": sha256((data / "old.md").read_bytes()).hexdigest()}
    assert datetime.fromisoformat(manifest["ingested_at"]).utcoffset().total_seconds() == 0
    assert manifest["embedding_model"] == ingest.EMBED_MODEL
    assert manifest["embedding_revision"] == ingest.embedding_versions()["embedding_revision"]
    assert manifest["sentence_transformers_version"] == ingest.embedding_versions()[
        "sentence_transformers_version"]
    assert manifest["previous_collection"] is None
    assert "Original runbook content" not in json.dumps(manifest)
    assert ingest.check_freshness(str(persist))["status"] == "current"
    ingest.ingest_documents(str(data), str(persist))
    assert ingest.check_freshness(str(persist))["corpus_sha256"] == manifest["corpus_sha256"]


def test_freshness_reports_added_modified_and_deleted_sources(index):
    data, persist, _ = index
    (data / "old.md").write_text("# Old\n\n" + "Changed runbook content. " * 3)
    (data / "added.md").write_text("# Added\n\n" + "Another runbook. " * 5)
    result = ingest.check_freshness(str(persist))
    assert result["status"] == "stale"
    assert result["modified"] == ["old.md"]
    assert result["added"] == ["added.md"]
    assert result["deleted"] == []
    (data / "old.md").unlink()
    assert ingest.check_freshness(str(persist))["deleted"] == ["old.md"]


def test_shortened_source_is_modified_rather_than_deleted(index):
    data, persist, _ = index
    (data / "old.md").write_text("short")
    result = ingest.check_freshness(str(persist))
    assert result["status"] == "stale"
    assert result["modified"] == ["old.md"]
    assert result["deleted"] == []


def test_freshness_reports_unavailable_directory_and_unreadable_file(index, monkeypatch):
    data, persist, _ = index
    assert ingest.check_freshness(str(persist), str(data / "missing"))["status"] == "unavailable"
    original_read = Path.read_bytes

    def unreadable(path):
        if path.name == "old.md":
            raise PermissionError("old.md is unreadable")
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", unreadable)
    result = ingest.check_freshness(str(persist))
    assert result["status"] == "unavailable"
    assert "old.md is unreadable" in result["reason"]


def test_unreadable_subdirectory_is_unavailable_rather_than_current(index, monkeypatch):
    data, persist, _ = index
    restricted = data / "restricted"
    restricted.mkdir()
    original_scandir = ingest.os.scandir

    def unreadable(path):
        if Path(path) == restricted:
            raise PermissionError("restricted directory is unreadable")
        return original_scandir(path)

    monkeypatch.setattr(ingest.os, "scandir", unreadable)
    result = ingest.check_freshness(str(persist))
    assert result["status"] == "unavailable"
    assert "restricted directory" in result["reason"]


def test_relocated_source_directory_can_be_checked_explicitly(index):
    data, persist, _ = index
    moved = data.with_name("moved")
    data.rename(moved)
    assert ingest.check_freshness(str(persist))["status"] == "unavailable"
    assert ingest.check_freshness(str(persist), str(moved))["status"] == "current"


@pytest.mark.parametrize("stale, expected_code", [(False, 0), (True, 1)])
def test_freshness_cli_does_not_load_embedder(index, monkeypatch, capsys, stale, expected_code):
    data, persist, _ = index
    if stale:
        (data / "old.md").unlink()
    monkeypatch.setattr("sentence_transformers.SentenceTransformer", Mock(side_effect=AssertionError))
    monkeypatch.setattr("sys.argv", ["ingest.py", "--check", "--persist-dir", str(persist)])
    capsys.readouterr()
    with pytest.raises(SystemExit) as exc:
        ingest.main()
    assert exc.value.code == expected_code
    assert json.loads(capsys.readouterr().out)["status"] == ("stale" if stale else "current")


def test_verbose_query_reports_snapshot_even_without_matches(index, capsys):
    _, _, reader = index
    reader.retrieve = Mock(return_value=[])
    reader.answer("irrelevant", verbose=True)
    output = capsys.readouterr().out
    manifest = json.loads(reader.collection.metadata["corpus_manifest"])
    assert manifest["ingested_at"] in output
    assert manifest["corpus_sha256"] in output
    assert ingest.EMBED_MODEL in output


def test_missing_index_cli_reports_unavailable_without_creating_it(tmp_path, monkeypatch, capsys):
    persist = tmp_path / "missing"
    monkeypatch.setattr("sys.argv", ["ingest.py", "--check", "--persist-dir", str(persist)])
    with pytest.raises(SystemExit) as exc:
        ingest.main()
    assert exc.value.code == 2
    assert json.loads(capsys.readouterr().out)["status"] == "unavailable"
    assert not persist.exists()


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    """Two-document index built with an embedder that records what it embeds."""
    embedded: list[str] = []

    def encode(texts, **_):
        embedded.extend(texts)
        return np.array([[1.0, float(len(text))] for text in texts])

    loader = Mock(return_value=Mock(encode=encode))
    monkeypatch.setattr("sentence_transformers.SentenceTransformer", loader)
    data = tmp_path / "data"
    data.mkdir()
    (data / "alpha.md").write_text("# Alpha\n\n" + "Alpha runbook content. " * 3)
    (data / "beta.md").write_text("# Beta\n\n" + "Beta runbook content. " * 3)
    persist = tmp_path / "index"
    ingest.ingest_documents(str(data), str(persist))
    embedded.clear()
    loader.reset_mock()
    return data, persist, embedded, loader


def _stored(persist):
    collection, _ = ingest._active_collection(str(persist))
    got = collection.get(include=["embeddings", "documents"])
    return {i: (doc, list(emb)) for i, doc, emb in zip(got["ids"], got["documents"], got["embeddings"], strict=False)}


def test_unchanged_reingest_embeds_nothing_and_keeps_active_index(corpus):
    data, persist, embedded, loader = corpus
    active = (persist / "active-collection").read_text()
    assert ingest.ingest_documents(str(data), str(persist)) == 2
    loader.assert_not_called()
    assert embedded == []
    assert (persist / "active-collection").read_text() == active


def test_editing_one_file_reembeds_only_that_file(corpus):
    data, persist, embedded, _ = corpus
    before = _stored(persist)
    (data / "beta.md").write_text("# Beta\n\n" + "Edited beta runbook content. " * 3)
    ingest.ingest_documents(str(data), str(persist))
    assert embedded == ["Beta\n\n" + ("Edited beta runbook content. " * 3).strip()]
    after = _stored(persist)
    assert after["alpha.md#0"] == before["alpha.md#0"]
    assert after["beta.md#0"][0].startswith("Beta\n\nEdited")
    assert ingest.check_freshness(str(persist))["status"] == "current"


def test_added_and_deleted_files_update_index_without_reembedding_others(corpus):
    data, persist, embedded, _ = corpus
    (data / "beta.md").unlink()
    (data / "gamma.md").write_text("# Gamma\n\n" + "Gamma runbook content. " * 3)
    assert ingest.ingest_documents(str(data), str(persist)) == 2
    assert len(embedded) == 1 and "Gamma" in embedded[0]
    assert set(_stored(persist)) == {"alpha.md#0", "gamma.md#0"}
    assert ingest.check_freshness(str(persist))["status"] == "current"


def test_deleting_only_a_file_switches_without_loading_embedder(corpus):
    data, persist, _, loader = corpus
    (data / "beta.md").unlink()
    assert ingest.ingest_documents(str(data), str(persist)) == 1
    loader.assert_not_called()
    assert set(_stored(persist)) == {"alpha.md#0"}


@pytest.mark.parametrize("kwargs", [{"rebuild": True}, {"chunk_size": 400}])
def test_rebuild_or_incompatible_index_reembeds_everything(corpus, kwargs):
    data, persist, embedded, _ = corpus
    ingest.ingest_documents(str(data), str(persist), **kwargs)
    assert len(embedded) == 2


def test_failed_incremental_update_keeps_previous_index(corpus, monkeypatch):
    data, persist, _, _ = corpus
    before = _stored(persist)
    active = (persist / "active-collection").read_text()
    (data / "beta.md").write_text("# Beta\n\n" + "Edited beta runbook content. " * 3)
    monkeypatch.setattr(Collection, "count", lambda self: 0)
    with pytest.raises(RuntimeError):
        ingest.ingest_documents(str(data), str(persist))
    monkeypatch.undo()
    assert (persist / "active-collection").read_text() == active
    assert _stored(persist) == before


def test_rebuild_cli_flag(monkeypatch):
    runner = Mock()
    monkeypatch.setattr(ingest, "ingest_documents", runner)
    monkeypatch.setattr("sys.argv", ["ingest.py", "--rebuild"])
    ingest.main()
    runner.assert_called_once_with("data", "chroma_db", 500, rebuild=True)


# ── Embedding stack versions (#62) ───────────────────────────────


def _installed(monkeypatch, st_version="6.1.0", revision="aaaa"):
    monkeypatch.setattr(ingest, "embedding_versions", lambda: {
        "sentence_transformers_version": st_version, "embedding_revision": revision})


@pytest.fixture
def versioned_corpus(corpus, monkeypatch):
    """The two-document corpus, re-indexed under a known embedding stack."""
    data, persist, embedded, loader = corpus
    _installed(monkeypatch)
    ingest.ingest_documents(str(data), str(persist), rebuild=True)
    embedded.clear()
    loader.reset_mock()
    return data, persist, embedded, loader


@pytest.mark.parametrize("installed", [{"st_version": "6.2.0"}, {"revision": "bbbb"}])
def test_embedding_version_change_reembeds_everything(versioned_corpus, monkeypatch, installed):
    data, persist, embedded, _ = versioned_corpus
    _installed(monkeypatch, **installed)
    result = ingest.check_freshness(str(persist))
    assert result["status"] == "stale"
    assert result["added"] == result["modified"] == result["deleted"] == []
    assert len(result["embedding_mismatch"]) == 1
    ingest.ingest_documents(str(data), str(persist))
    assert len(embedded) == 2
    manifest = ingest._active_collection(str(persist))[1]
    assert manifest["sentence_transformers_version"] == installed.get("st_version", "6.1.0")
    assert manifest["embedding_revision"] == installed.get("revision", "aaaa")
    assert ingest.check_freshness(str(persist))["status"] == "current"


def test_unknown_cached_revision_does_not_force_reembed(versioned_corpus, monkeypatch):
    """Without a resolvable local snapshot only the package version is compared."""
    data, persist, embedded, loader = versioned_corpus
    _installed(monkeypatch, revision=None)
    assert ingest.check_freshness(str(persist))["status"] == "current"
    ingest.ingest_documents(str(data), str(persist))
    loader.assert_not_called()
    assert embedded == []
    (data / "gamma.md").write_text("# Gamma\n\n" + "Gamma runbook content. " * 3)
    ingest.ingest_documents(str(data), str(persist))
    assert len(embedded) == 1
    assert ingest._active_collection(str(persist))[1]["embedding_revision"] == "aaaa"


def _legacy_index(persist):
    """Make active an index written before versions and history were recorded."""
    manifest = {
        "schema_version": 1, "ingested_at": "2026-01-01T00:00:00+00:00",
        "data_dir": "/nowhere", "embedding_model": ingest.EMBED_MODEL,
        "embedding_revision": None, "chunk_size": 500, "sources": {"alpha.md": "0" * 64},
        "corpus_sha256": "1" * 64,
    }
    name = "observability-docs-" + "0" * 32
    client = chromadb.PersistentClient(path=str(persist))
    collection = client.create_collection(
        name, metadata={"hnsw:space": "cosine", "corpus_manifest": json.dumps(manifest)})
    collection.add(ids=["alpha.md#0"], embeddings=[[1.0, 0.0]], documents=["Alpha legacy runbook"],
                   metadatas=[{"source": "alpha.md", "file": "alpha.md", "section": "Alpha",
                               "source_sha256": "0" * 64}])
    (persist / "active-collection").write_text(name)
    return name


def test_legacy_manifest_stays_queryable_and_next_ingest_reembeds(index, monkeypatch):
    data, persist, _ = index
    legacy = _legacy_index(persist)
    reader = RAGEngine(str(persist))
    assert reader.collection.name == legacy
    assert [c["source"] for c in reader.retrieve("runbook")] == ["alpha.md"]
    result = ingest.check_freshness(str(persist), str(data))
    assert result["status"] == "stale"
    assert "not recorded" in result["embedding_mismatch"][0]
    encode = Mock(return_value=np.array([[1.0, 0.0]]))
    monkeypatch.setattr("sentence_transformers.SentenceTransformer",
                        Mock(return_value=Mock(encode=encode)))
    ingest.ingest_documents(str(data), str(persist))
    encode.assert_called_once()
    manifest = ingest._active_collection(str(persist))[1]
    assert manifest["sentence_transformers_version"]
    assert manifest["previous_collection"] == legacy
    assert ingest.check_freshness(str(persist))["status"] == "current"


# ── Pruning superseded collections (#61) ─────────────────────────


def _names(persist):
    return {c.name for c in chromadb.PersistentClient(path=str(persist)).list_collections()}


def _revise(data, persist, times):
    """Ingest `times` edits, returning the active collection after each."""
    actives = []
    for i in range(times):
        (data / "old.md").write_text(f"# Old\n\nRevision {i} of the runbook content. " * 3)
        ingest.ingest_documents(str(data), str(persist))
        actives.append((persist / "active-collection").read_text())
    return actives


def test_prune_keeps_active_and_previous_by_default(index, capsys):
    data, persist, _ = index
    first = (persist / "active-collection").read_text()
    *_, previous, active = [first, *_revise(data, persist, 3)]
    assert "run `python ingest.py --prune`" in capsys.readouterr().out
    result = ingest.prune_collections(str(persist))
    assert result["active"] == active
    assert result["kept"] == [active, previous]
    assert len(result["removed"]) == 2 and first in result["removed"]
    assert _names(persist) == {active, previous}
    assert ingest.check_freshness(str(persist))["status"] == "current"
    assert RAGEngine(str(persist)).retrieve("runbook")
    # Rolling back to the kept previous collection works.
    (persist / "active-collection").write_text(previous)
    assert RAGEngine(str(persist)).collection.name == previous
    assert ingest.prune_collections(str(persist))["removed"] == []


def test_prune_keep_is_configurable_and_never_removes_active(index):
    data, persist, _ = index
    actives = _revise(data, persist, 3)
    assert ingest.prune_collections(str(persist), keep=3)["kept"] == actives[::-1]
    with pytest.raises(ValueError):
        ingest.prune_collections(str(persist), keep=0)
    # After a rollback, the pointer, not recency, decides what is active.
    (persist / "active-collection").write_text(actives[0])
    result = ingest.prune_collections(str(persist), keep=1)
    assert _names(persist) == {actives[0]}
    assert result["removed"] == sorted(actives[1:])
    assert RAGEngine(str(persist)).retrieve("runbook")


def test_prune_removes_incomplete_build_and_keeps_previous(index, monkeypatch):
    data, persist, _ = index
    first = (persist / "active-collection").read_text()
    active = _revise(data, persist, 1)[0]
    (data / "old.md").write_text("# Old\n\n" + "Failed revision content. " * 3)
    monkeypatch.setattr(Collection, "count", lambda self: 0)
    with pytest.raises(RuntimeError):
        ingest.ingest_documents(str(data), str(persist))
    monkeypatch.undo()
    assert len(_names(persist)) == 3
    result = ingest.prune_collections(str(persist))
    assert _names(persist) == {active, first}
    assert len(result["removed"]) == 1


def test_prune_dry_run_and_unmanaged_collections(index):
    data, persist, _ = index
    _revise(data, persist, 2)
    chromadb.PersistentClient(path=str(persist)).create_collection("someone-elses-data")
    before = _names(persist)
    result = ingest.prune_collections(str(persist), dry_run=True)
    assert result["removed"] == [] and len(result["would_remove"]) == 1
    assert _names(persist) == before
    ingest.prune_collections(str(persist), keep=1)
    assert "someone-elses-data" in _names(persist)


def _segment_dirs(persist):
    return {p.name for p in persist.iterdir() if p.is_dir()}


def test_prune_removes_orphaned_vector_files_only(index):
    data, persist, _ = index
    _revise(data, persist, 2)
    before = _segment_dirs(persist)
    unrelated = persist / "00000000-0000-0000-0000-000000000000"
    unrelated.mkdir()
    (unrelated / "notes.txt").write_text("not an index")
    result = ingest.prune_collections(str(persist), keep=1)
    assert result["segment_dirs_removed"] and result["bytes_freed"] > 0
    assert set(result["segment_dirs_removed"]) <= before
    assert unrelated.exists()
    assert ingest.prune_collections(str(persist), keep=1)["segment_dirs_removed"] == []
    assert RAGEngine(str(persist)).retrieve("runbook")
    assert ingest.check_freshness(str(persist))["status"] == "current"


def test_prune_without_history_keeps_newest_others(index):
    data, persist, _ = index
    first = (persist / "active-collection").read_text()
    newest = _revise(data, persist, 1)[0]
    legacy = _legacy_index(persist)  # active, with no recorded predecessor
    result = ingest.prune_collections(str(persist))
    assert result["kept"] == [legacy, newest]
    assert result["removed"] == [first]


def test_prune_refuses_when_active_collection_is_missing(index):
    data, persist, _ = index
    _revise(data, persist, 1)
    before = _names(persist)
    (persist / "active-collection").write_text("observability-docs-" + "f" * 32)
    with pytest.raises(ValueError, match="not found"):
        ingest.prune_collections(str(persist))
    assert _names(persist) == before


def test_prune_cli_reports_removed(index, monkeypatch, capsys):
    data, persist, _ = index
    _revise(data, persist, 2)
    monkeypatch.setattr("sentence_transformers.SentenceTransformer",
                        Mock(side_effect=AssertionError))
    monkeypatch.setattr("sys.argv", ["ingest.py", "--prune", "--keep", "1",
                                     "--persist-dir", str(persist)])
    capsys.readouterr()
    ingest.main()
    assert len(json.loads(capsys.readouterr().out)["removed"]) == 2
