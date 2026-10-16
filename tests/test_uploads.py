"""Validate uploads and the shared ingestion path without downloading models."""

from unittest.mock import Mock

import pytest

import ingest
from ingest import ingest_documents, load_documents, save_uploaded_documents

DOCUMENT = b"# Database Runbook\n\n## Latency\n\nCheck slow queries and database CPU before escalating."


@pytest.mark.parametrize("name, content", [
    ("../escape.md", DOCUMENT),
    ("/absolute.md", DOCUMENT),
    ("nested/file.md", DOCUMENT),
    ("nested\\file.md", DOCUMENT),
    ("null\x00.md", DOCUMENT),
    ("x" * 253 + ".md", DOCUMENT),
    ("file.pdf", DOCUMENT),
    ("file.md", b"too short"),
    ("file.md", b"\xff" * 60),
    ("file.md", b"\x00" * 60),
    ("file.md", b"# " + b"heading" * 10),
    ("file.md", b"x" * (5 * 1024 * 1024 + 1)),
])
def test_invalid_batch_writes_nothing(tmp_path, name, content):
    with pytest.raises(ValueError):
        save_uploaded_documents([("valid.md", DOCUMENT), (name, content)], str(tmp_path))
    assert not list(tmp_path.iterdir())


def test_uploads_preserve_same_name_versions_and_deduplicate_retries(tmp_path):
    original = save_uploaded_documents([("runbook.md", DOCUMENT)], str(tmp_path))[0]
    retry = save_uploaded_documents([("runbook.md", DOCUMENT)], str(tmp_path))[0]
    replacement = save_uploaded_documents(
        [("runbook.md", DOCUMENT + b"\nContact the database team.")], str(tmp_path),
    )[0]
    assert original == retry
    assert original != replacement
    assert original.read_bytes() == DOCUMENT
    assert len(load_documents(str(tmp_path))) == 2


def test_upload_normalizes_bom_and_uppercase_extension(tmp_path):
    path = save_uploaded_documents([("Runbook.MD", b"\xef\xbb\xbf" + DOCUMENT)], str(tmp_path))[0]
    assert path.name == "Runbook.md"
    assert path.read_bytes() == DOCUMENT
    assert len(load_documents(str(tmp_path))) == 1


@pytest.mark.parametrize("content", [None, "# " + "heading" * 10])
def test_empty_corpus_does_not_touch_index(tmp_path, monkeypatch, content):
    if content:
        (tmp_path / "heading.md").write_text(content)
    client = Mock()
    monkeypatch.setattr(ingest.chromadb, "PersistentClient", client)
    with pytest.raises(ValueError):
        ingest_documents(str(tmp_path), str(tmp_path / "index"))
    client.assert_not_called()


def test_shared_ingestion_stores_uploads_with_source_and_section(tmp_path, monkeypatch):
    path = save_uploaded_documents([("runbook.md", DOCUMENT)], str(tmp_path))[0]
    embedder = Mock()
    embedder.encode.return_value.tolist.return_value = [[0.1, 0.2]]
    monkeypatch.setattr("sentence_transformers.SentenceTransformer", Mock(return_value=embedder))
    client = Mock()
    monkeypatch.setattr(ingest.chromadb, "PersistentClient", Mock(return_value=client))
    assert ingest_documents(str(tmp_path), str(tmp_path / "index")) == 1
    stored = client.create_collection.return_value.add.call_args.kwargs
    source = path.relative_to(tmp_path).as_posix()
    assert stored["ids"] == [f"{source}#0"]
    assert stored["metadatas"] == [{"source": source, "file": "runbook.md", "section": "Latency"}]
    assert "Check slow queries" in stored["documents"][0]


def test_cli_uses_shared_ingestion(monkeypatch):
    runner = Mock()
    monkeypatch.setattr(ingest, "ingest_documents", runner)
    monkeypatch.setattr("sys.argv", ["ingest.py", "--data-dir", "docs", "--persist-dir", "index"])
    ingest.main()
    runner.assert_called_once_with("docs", "index", 500)
