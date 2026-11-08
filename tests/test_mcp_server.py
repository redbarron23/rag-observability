"""Read-only MCP server, driven through the SDK's in-process client.

The bundled corpus in data/ is indexed with a fixed fake embedder, so no model
download and no provider key is needed. Every result is checked against the
same closed contract schemas the HTTP API tests use.
"""

import json
from pathlib import Path
from unittest.mock import Mock

import anyio
import numpy as np
import pytest
from contract_check import ERROR, REQUEST, RETRIEVE, load_schema
from contract_check import errors as contract_errors
from mcp import Client

import api
import ingest
import mcp_server
from query import RAGEngine

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def fake_embedder():
    """Texts mentioning "tag" embed as [1, 0], everything else as [0, 1]."""
    def encode(texts, **_):
        return np.array([[1.0, 0.0] if "tag" in t.lower() else [0.0, 1.0] for t in texts])
    return Mock(encode=Mock(side_effect=encode))


@pytest.fixture
def bundled_index(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr("sentence_transformers.SentenceTransformer",
                        Mock(return_value=fake_embedder()))
    persist = tmp_path / "index"
    ingest.ingest_documents(str(DATA_DIR), str(persist))
    return persist


def service_for(persist, **kwargs):
    return api.RetrievalService(str(persist), embedder_factory=fake_embedder, **kwargs)


async def _session(service, action):
    async with Client(mcp_server.create_server(service)) as client:
        return await action(client)


def run(service, action):
    return anyio.run(_session, service, action)


def call(service, arguments):
    return run(service, lambda client: client.call_tool("retrieve", arguments))


def error_body(result):
    """A tool error carries the HTTP error body as JSON text."""
    assert result.is_error is True
    body = json.loads(result.content[0].text)
    assert not contract_errors(ERROR, body), contract_errors(ERROR, body)
    assert body["error"]["code"] in load_schema(ERROR)["$defs"]["known_error_code"]["enum"]
    return body


def test_retrieve_tool_is_listed_with_the_contract_schemas(bundled_index):
    result = run(service_for(bundled_index), lambda client: client.list_tools())
    assert [tool.name for tool in result.tools] == ["retrieve"]
    tool = result.tools[0]
    # Same request and response contract as POST /retrieve, not a second format.
    assert tool.input_schema == load_schema(REQUEST)
    assert tool.output_schema == load_schema(RETRIEVE)
    assert tool.annotations.read_only_hint is True
    assert tool.annotations.destructive_hint is False
    assert "UNTRUSTED" in tool.description and "injected instructions" in tool.description


def test_call_against_bundled_corpus_matches_the_response_contract(bundled_index, monkeypatch):
    generate = Mock(side_effect=AssertionError("retrieval must not generate"))
    monkeypatch.setattr(RAGEngine, "_call_llm", generate)
    monkeypatch.setattr(RAGEngine, "_call_anthropic", generate)
    service = service_for(bundled_index)

    result = call(service, {"question": "Which tags are required?", "k": 3, "max_distance": 0.5})

    assert result.is_error is False
    data = result.structured_content
    assert not contract_errors(RETRIEVE, data), contract_errors(RETRIEVE, data)
    assert json.loads(result.content[0].text) == data  # text mirror for older clients
    assert data["generated"] is False and data["content_trust"] == "untrusted"
    assert data["query"] == {"question": "Which tags are required?", "k": 3,
                             "max_distance": 0.5, "retrieval": "vector"}
    # Only ranked hits must match. The fake embedder ties every chunk containing "tag"
    # (including "percentages"), so which tied chunks rank is arbitrary, and section
    # completion then appends their siblings (expanded_from set), which need not match.
    ranked = [hit for hit in data["hits"] if hit["expanded_from"] is None]
    assert len(ranked) == 3 and all("tag" in hit["text"].lower() for hit in ranked)
    assert data["corpus"]["provenance"] == "available"
    assert data["corpus"]["document_count"] == len(list(DATA_DIR.rglob("*.md")))
    generate.assert_not_called()

    # The same service answers the HTTP route identically (modulo checked_at).
    direct = service.retrieve({"question": "Which tags are required?", "k": 3,
                               "max_distance": 0.5})
    for body in (data, direct):
        body["freshness"].pop("checked_at")
    assert direct == data


def test_defaults_and_hybrid_mode_are_reported(bundled_index):
    data = call(service_for(bundled_index, retrieval="hybrid"),
                {"question": "tagging"}).structured_content
    assert data["query"]["k"] == 6 and data["query"]["max_distance"] == 0.6
    assert data["query"]["retrieval"] == "hybrid"
    assert not contract_errors(RETRIEVE, data)


def test_no_match_is_an_empty_success_not_an_error(bundled_index):
    service = api.RetrievalService(
        str(bundled_index),
        embedder_factory=lambda: Mock(encode=Mock(return_value=np.array([[-1.0, -1.0]]))))
    result = call(service, {"question": "Nothing relevant", "max_distance": 0.1})
    assert result.is_error is False and result.structured_content["hits"] == []


@pytest.mark.parametrize("arguments, message", [
    ({}, "'question' must be a string"),
    ({"question": "   "}, "must not be empty"),
    ({"question": 5}, "must be a string"),
    ({"question": "x" * (api.MAX_QUESTION_CHARS + 1)}, "at most"),
    ({"question": "a\x00b"}, "NUL"),
    ({"question": "q", "k": 0}, "'k'"),
    ({"question": "q", "k": 2.0}, "'k'"),
    ({"question": "q", "k": True}, "'k'"),
    ({"question": "q", "max_distance": 2.5}, "'max_distance'"),
    ({"question": "q", "max_distance": "0.5"}, "'max_distance'"),
    ({"question": "q", "n_results": 3}, "Unknown field"),
])
def test_invalid_arguments_are_tool_errors_like_http_400(bundled_index, arguments, message):
    body = error_body(call(service_for(bundled_index), arguments))
    assert body["error"]["code"] == "invalid_request" and message in body["error"]["message"]


def test_missing_index_is_a_tool_error_and_recovers_once_ingested(tmp_path, monkeypatch):
    monkeypatch.setattr("sentence_transformers.SentenceTransformer",
                        Mock(return_value=fake_embedder()))
    persist = tmp_path / "index"
    service = service_for(persist)

    body = error_body(call(service, {"question": "tags"}))
    assert body["error"]["code"] == "index_unavailable"
    assert "ingest.py" in body["error"]["message"]
    assert not persist.exists()  # a query must not create an empty index

    ingest.ingest_documents(str(DATA_DIR), str(persist))
    assert call(service, {"question": "tags"}).is_error is False


def test_require_fresh_reports_stale_index_with_freshness(tmp_path, monkeypatch):
    monkeypatch.setattr("sentence_transformers.SentenceTransformer",
                        Mock(return_value=fake_embedder()))
    data = tmp_path / "data"
    data.mkdir()
    (data / "tagging.md").write_text("# Tags\n\n## Required\n\n" + "Every resource needs tags. " * 3)
    persist = tmp_path / "index"
    ingest.ingest_documents(str(data), str(persist))
    (data / "new.md").write_text("# New\n\n" + "Fresh content about tags. " * 3)

    body = error_body(call(service_for(persist, data_dir=str(data), require_fresh=True),
                           {"question": "tags"}))
    assert body["error"]["code"] == "index_stale"
    assert body["error"]["freshness"]["added"] == ["new.md"]


def test_engine_failure_is_a_generic_tool_error(bundled_index, monkeypatch):
    monkeypatch.setattr(RAGEngine, "retrieve", Mock(side_effect=RuntimeError("secret path")))
    body = error_body(call(service_for(bundled_index), {"question": "tags"}))
    assert body["error"]["code"] == "retrieval_failed"
    assert "secret" not in json.dumps(body)


def test_unknown_tool_is_an_error(bundled_index):
    result = run(service_for(bundled_index), lambda client: client.call_tool("generate", {}))
    assert error_body(result)["error"]["code"] == "not_found"
