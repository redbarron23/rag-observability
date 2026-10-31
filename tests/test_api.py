"""HTTP contracts with a stub engine: no model downloads or provider calls."""

import json
from unittest.mock import Mock

import pytest
from chromadb.errors import NotFoundError
from fastapi.testclient import TestClient

import api

CHUNK = {
    "id": "reference/coverage-targets.md#1", "text": "Tier 1 target: 85% within 4 months.",
    "source": "reference/coverage-targets.md", "file": "coverage-targets.md",
    "section": "Tier 1", "distance": 0.25,
}
MANIFEST = {
    "schema_version": 1, "corpus_sha256": "abc123", "ingested_at": "2026-10-31T12:00:00Z",
    "embedding_model": "all-MiniLM-L6-v2", "data_dir": "/private/docs",
    "sources": {"reference/coverage-targets.md": "source-hash"},
}
CITATION = {"source": CHUNK["source"], "section": "Tier 1", "chunk": 1,
            "cited_text": "85% within 4 months"}


@pytest.fixture
def client(monkeypatch):
    engine = Mock()
    engine.collection.metadata = {"corpus_manifest": json.dumps(MANIFEST)}
    engine.retrieve.return_value = [CHUNK]
    engine.answer_with_citations.return_value = ("85% within 4 months.", [CHUNK], [CITATION])
    factory = Mock(return_value=engine)
    monkeypatch.setattr(api, "RAGEngine", factory)
    monkeypatch.setenv("RAG_PERSIST_DIR", "/selected/index")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with TestClient(api.app) as http:
        yield http, engine, factory
    assert api.app.state.engine is None


def test_retrieve_returns_evidence_and_reuses_engine_without_generation(client):
    http, engine, factory = client
    for _ in range(2):
        response = http.post("/retrieve", json={"question": "  Tier 1 target?  "})
        assert response.status_code == 200
        data = response.json()
        assert data["chunks"] == [CHUNK]
        assert data["corpus"] == {
            "status": "available", "corpus_sha256": "abc123",
            "ingested_at": MANIFEST["ingested_at"], "embedding_model": MANIFEST["embedding_model"],
        }
        assert "/private/docs" not in response.text
    factory.assert_called_once_with(persist_dir="/selected/index")
    engine.retrieve.assert_called_with("Tier 1 target?", n_results=6, max_distance=0.6)
    engine.answer.assert_not_called()
    engine.answer_with_citations.assert_not_called()
    engine._call_llm.assert_not_called()


def test_ask_returns_answer_chunks_and_structured_citations(client):
    http, engine, _ = client
    response = http.post("/ask", json={"question": "Tier 1 target?", "n_results": 3,
                                       "max_distance": 0.4})
    assert response.status_code == 200
    assert response.json()["answer"] == "85% within 4 months."
    assert response.json()["chunks"] == [CHUNK]
    assert response.json()["citations"] == [CITATION]
    engine.answer_with_citations.assert_called_once_with(
        "Tier 1 target?", n_results=3, max_distance=0.4,
    )


def test_prompt_based_citations_and_no_matches_are_preserved(client):
    http, engine, _ = client
    engine.answer_with_citations.return_value = ("85% (source: doc.md).", [CHUNK], None)
    assert http.post("/ask", json={"question": "target?"}).json()["citations"] is None
    engine.retrieve.return_value = []
    response = http.post("/retrieve", json={"question": "unrelated?"})
    assert response.status_code == 200
    assert response.json()["chunks"] == []


@pytest.mark.parametrize("payload", [
    {}, {"question": ""}, {"question": " \n "}, {"question": "x" * 4001},
    {"question": 42}, {"question": "q", "n_results": 0},
    {"question": "q", "n_results": 21}, {"question": "q", "n_results": 1.5},
    {"question": "q", "n_results": True}, {"question": "q", "max_distance": -0.1},
    {"question": "q", "max_distance": 2.1}, {"question": "q", "max_distance": "NaN"},
    {"question": "q", "persist_dir": "/another/index"},
])
@pytest.mark.parametrize("endpoint", ["/retrieve", "/ask"])
def test_invalid_requests_do_not_query_engine(client, payload, endpoint):
    http, engine, _ = client
    assert http.post(endpoint, json=payload).status_code == 422
    engine.retrieve.assert_not_called()
    engine.answer_with_citations.assert_not_called()


@pytest.mark.parametrize("metadata", [None, {}, {"corpus_manifest": "broken"},
                                       {"corpus_manifest": "[]"},
                                       {"corpus_manifest": '{"schema_version": 2}'},
                                       {"corpus_manifest": '{"schema_version": 1}'}])
def test_legacy_or_invalid_provenance_is_explicit(client, metadata):
    http, engine, _ = client
    engine.collection.metadata = metadata
    response = http.post("/retrieve", json={"question": "target?"})
    assert response.status_code == 200
    assert response.json()["corpus"] == {"status": "unavailable"}


@pytest.mark.parametrize("endpoint", ["/retrieve", "/ask"])
def test_missing_index_reports_unavailable_instead_of_no_matches(monkeypatch, endpoint):
    factory = Mock(side_effect=NotFoundError("private filesystem detail"))
    monkeypatch.setattr(api, "RAGEngine", factory)
    with TestClient(api.app) as http:
        response = http.post(endpoint, json={"question": "target?"})
    assert response.status_code == 503
    assert "python ingest.py" in response.json()["detail"]
    assert "private filesystem detail" not in response.text


@pytest.mark.parametrize("endpoint,method,status", [
    ("/retrieve", "retrieve", 503), ("/ask", "answer_with_citations", 502),
])
def test_query_failures_do_not_expose_internal_details(client, endpoint, method, status):
    http, engine, _ = client
    getattr(engine, method).side_effect = RuntimeError("secret provider detail")
    response = http.post(endpoint, json={"question": "target?"})
    assert response.status_code == status
    assert "secret provider detail" not in response.text


def test_index_disappearing_during_generation_is_unavailable(client):
    http, engine, _ = client
    engine.answer_with_citations.side_effect = NotFoundError("collection gone")
    assert http.post("/ask", json={"question": "target?"}).status_code == 503
