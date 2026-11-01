"""Eval run artifacts (issue #39): schema, secrets and consistency, offline.

The engine and the provider clients are fakes, so no model or API is used.
"""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import eval_artifact
import evals
from query import RAGEngine

SECRET = "sk-ant-test-0123456789abcdef"
DOC_TEXT = "CONFIDENTIAL-CHUNK-TEXT: the Tier 1 target is 85% within 4 months."
ANSWER = "The Tier 1 target is 85% within 4 months (source: doc.md, section 'Tier 1')."
MANIFEST = {"schema_version": 1, "corpus_sha256": "ab" * 32,
            "ingested_at": "2026-11-01T00:00:00+00:00", "embedding_model": "all-MiniLM-L6-v2",
            "embedding_revision": None, "chunk_size": 500, "data_dir": "/private/path",
            "sources": {"doc.md": {"sha256": "cd" * 32}}}
TOP_LEVEL_KEYS = {"schema_version", "kind", "run", "code", "corpus", "embedding",
                  "retrieval", "generation", "results", "heldout"}


class FakeEngine:
    """Answers every question from one chunk; raises for questions containing "boom"."""

    collection_name = "observability-docs-test"
    last_generation = None

    def __init__(self):
        self.collection = Mock()
        self.collection.metadata = {"corpus_manifest": json.dumps(MANIFEST)}
        self.collection.count.return_value = 1
        self.collection.get.return_value = {"documents": [DOC_TEXT]}

    def retrieve(self, question, n_results=6, max_distance=0.6):
        if "unanswerable" in question:
            return []
        return [{"id": "doc.md#0", "source": "doc.md", "section": "Tier 1",
                 "text": DOC_TEXT, "distance": 0.2}]

    def answer_with_citations(self, question, **kwargs):
        self.last_generation = {"llm_called": True, "provider": "anthropic",
                                "requested_model": "claude-sonnet-5-5"}
        if "boom" in question:
            raise RuntimeError(f"provider rejected key {SECRET}")
        self.last_generation.update(
            response_model="claude-sonnet-5-5", stop_reason="end_turn",
            usage={"input_tokens": 100, "output_tokens": 20}, fallback_used=False)
        return ANSWER, self.retrieve(question), None

    def answer(self, question, **kwargs):
        self.last_generation = {"llm_called": False,
                                "reason": "no chunks retrieved, so no provider call"}
        return "No relevant documents found.", []


@pytest.fixture
def run_evals(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    case = {"expected_sources": ["doc.md"], "expected_sections": [("doc.md", "Tier 1")],
            "expected_facts": ["85%"]}
    monkeypatch.setattr(evals, "EVALS", [
        {"id": "main-ok", "question": "Tier 1?", **case},
        {"id": "main-error", "question": "boom", **case},
    ])
    monkeypatch.setattr(evals, "UNANSWERABLE_EVALS",
                        [{"id": "main-decline", "question": "unanswerable rota?"}])
    monkeypatch.setattr(evals, "HELDOUT_EVALS", [{"id": "heldout-ok", "question": "T1?", **case}])
    monkeypatch.setattr(evals, "HELDOUT_UNANSWERABLE_EVALS",
                        [{"id": "heldout-decline", "question": "unanswerable owner?"}])
    monkeypatch.setattr(evals, "RAGEngine", lambda **kwargs: FakeEngine())

    def run(*flags):
        out = tmp_path / "out" / "run.json"
        monkeypatch.setattr("sys.argv", ["evals.py", "--output", str(out), *flags])
        with pytest.raises(SystemExit) as exc:
            evals.main()
        text = out.read_text(encoding="utf-8")
        return exc.value.code, text, json.loads(text)
    return run


def case_ids(block):
    return [c["id"] for c in block["cases"]]


def test_retrieval_only_artifact_schema_and_held_out_separation(run_evals):
    code, text, artifact = run_evals()

    assert set(artifact) == TOP_LEVEL_KEYS
    assert artifact["schema_version"] == eval_artifact.SCHEMA_VERSION
    assert artifact["run"]["exit_code"] == code == 0
    assert artifact["corpus"]["corpus_sha256"] == MANIFEST["corpus_sha256"]
    assert "data_dir" not in artifact["corpus"]  # no local paths
    # Unknown metadata is null with a reason, never a guess.
    assert artifact["embedding"]["indexed"]["revision"] is None
    assert artifact["embedding"]["indexed_revision_unavailable_reason"]
    assert artifact["generation"]["provider"] is None
    assert artifact["generation"]["unavailable_reason"]
    assert artifact["results"]["full"] == artifact["heldout"]["full"] == {
        "not_run_reason": "retrieval-only run (no --with-llm)"}
    assert artifact["retrieval"]["mode"] == "vector"

    retrieval = artifact["results"]["retrieval"]
    assert case_ids(retrieval) == ["main-ok", "main-error", "main-decline"]
    assert case_ids(artifact["heldout"]["retrieval"]) == ["heldout-ok", "heldout-decline"]
    assert artifact["heldout"]["counts_toward_exit_code"] is False
    assert retrieval["summary"]["total"] == 3
    assert retrieval["summary"]["by_dimension"]["sources"] == {
        "passed": 2, "failed": 0, "unevaluated": 0}
    assert all(c["duration_seconds"] >= 0 for c in retrieval["cases"])

    assert DOC_TEXT not in text and "CONFIDENTIAL" not in text
    assert SECRET not in text


def test_full_artifact_records_errors_usage_and_returned_model(run_evals):
    code, text, artifact = run_evals("--with-llm")

    assert code == artifact["run"]["exit_code"] == 1  # main-error fails the run
    full = artifact["results"]["full"]
    by_id = {c["id"]: c for c in full["cases"]}
    error = by_id["main-error"]
    assert error["status"] == "error"
    assert error["dimensions"] == {"sources": None, "facts": None, "citations": None}
    assert error["error"]["type"] == "RuntimeError"
    assert eval_artifact.REDACTED in error["error"]["message"]
    assert by_id["main-ok"]["generation"]["usage"] == {"input_tokens": 100, "output_tokens": 20}
    assert by_id["main-decline"]["generation"]["llm_called"] is False
    assert full["summary"]["provider_errors"] == 1
    assert full["summary"]["by_dimension"]["facts"] == {"passed": 1, "failed": 0, "unevaluated": 1}
    assert full["summary"]["usage"] == {"provider_calls": 2, "calls_with_usage": 1,
                                        "tokens": {"input_tokens": 100, "output_tokens": 20}}
    assert artifact["generation"]["requested_model"] == "claude-sonnet-5-5"
    assert artifact["generation"]["response_models"] == ["claude-sonnet-5-5"]
    # Held-out cases stay out of the main results and the exit code.
    assert case_ids(artifact["heldout"]["full"]) == ["heldout-ok", "heldout-decline"]
    assert not {"heldout-ok", "heldout-decline"} & set(by_id)

    assert SECRET not in text
    assert DOC_TEXT not in text
    assert ANSWER not in text and all("answer" not in c for c in full["cases"])


def test_answers_are_recorded_only_on_request(run_evals):
    _, text, artifact = run_evals("--with-llm", "--include-answers")
    assert artifact["run"]["include_answers"] is True
    by_id = {c["id"]: c for c in artifact["results"]["full"]["cases"]}
    assert by_id["main-ok"]["answer"] == ANSWER
    assert SECRET not in text


def test_redact_only_replaces_configured_key_values(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek-secret-value")
    assert eval_artifact.redact("bad key sk-deepseek-secret-value") == "bad key [REDACTED]"
    assert eval_artifact.redact(None) is None


# ── query.py records what the provider returned ────────────────────────

CHUNKS = [{"source": "doc.md", "section": "Tier 1", "text": "85% within 4 months."}]


def test_anthropic_call_records_returned_model_and_usage(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    client = Mock()
    client.beta.messages.create.return_value = SimpleNamespace(
        stop_reason="end_turn", model="claude-opus-5-5",
        content=[SimpleNamespace(type="text", text="85%.", citations=None)],
        usage=SimpleNamespace(input_tokens=2100, output_tokens=280,
                              cache_creation_input_tokens=0, cache_read_input_tokens=None,
                              iterations=[SimpleNamespace(type="message"),
                                          SimpleNamespace(type="fallback_message")]),
    )
    monkeypatch.setattr("anthropic.Anthropic", Mock(return_value=client))
    engine = object.__new__(RAGEngine)

    assert engine._call_llm("q", CHUNKS) == ("85%.", [])  # return shape unchanged
    assert engine.last_generation == {
        "llm_called": True, "provider": "anthropic", "requested_model": "claude-sonnet-5-5",
        "response_model": "claude-opus-5-5", "stop_reason": "end_turn",
        "usage": {"input_tokens": 2100, "output_tokens": 280, "cache_creation_input_tokens": 0},
        "fallback_used": True,
    }


def test_deepseek_call_records_usage_in_common_field_names(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    client = Mock()
    client.chat.completions.create.return_value = SimpleNamespace(
        model="deepseek-chat",
        choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content="85%."))],
        usage=SimpleNamespace(prompt_tokens=900, completion_tokens=50),
    )
    monkeypatch.setattr("openai.OpenAI", Mock(return_value=client))
    engine = object.__new__(RAGEngine)

    assert engine._call_llm("q", CHUNKS) == ("85%.", None)
    assert engine.last_generation["response_model"] == "deepseek-chat"
    assert engine.last_generation["usage"] == {"input_tokens": 900, "output_tokens": 50}
    assert engine.last_generation["fallback_used"] is None


def test_missing_usage_stays_null():
    assert eval_artifact.usage_totals([{"generation": {"llm_called": True, "usage": None}}]) == {
        "provider_calls": 1, "calls_with_usage": 0, "tokens": {}}
