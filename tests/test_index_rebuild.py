"""Replacement failures must leave both existing and new readers usable."""

from unittest.mock import Mock

import chromadb
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
