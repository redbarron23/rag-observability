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
    assert manifest["embedding_revision"] is None
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
