"""Trace the production engine with deterministic retrieval and mocked providers."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

import query

QUESTION = "PRIVATE QUESTION"
DOCUMENT = "PRIVATE DOCUMENT"
ANSWER = "PRIVATE ANSWER"


@pytest.fixture(autouse=True)
def tracing_off(monkeypatch):
    monkeypatch.delenv("OTEL_TRACES_EXPORTER", raising=False)
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
    query._trace_provider.cache_clear()
    yield
    query._trace_provider.cache_clear()


@pytest.fixture
def spans(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider(shutdown_on_exit=False)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(query, "_tracer", lambda: provider.get_tracer("test"))
    yield exporter.get_finished_spans
    provider.shutdown()


@pytest.fixture
def engine(monkeypatch):
    eng = object.__new__(query.RAGEngine)
    eng.retrieval = "vector"
    eng.embedder = Mock()
    eng.embedder.encode.return_value.tolist.return_value = [[1.0, 0.0]]
    eng._retrieve_vector = Mock(return_value=[{
        "id": "fictional-chunk", "file": "policy.md", "source": "fictional/policy.md",
        "distance": 0.2, "text": DOCUMENT, "section": "Policy",
    }])
    eng._complete_sections = lambda embedding, chunks: chunks
    eng.policy_statuses = lambda day: None
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    response = SimpleNamespace(
        model="returned-model", stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=12, output_tokens=4,
                              iterations=[SimpleNamespace(type="fallback_message")]),
        content=[SimpleNamespace(type="text", text=ANSWER, citations=[])],
    )
    client = Mock()
    client.beta.messages.create.return_value = response
    monkeypatch.setattr("anthropic.Anthropic", Mock(return_value=client))
    return eng, client, response


def test_question_has_retrieval_and_generation_children(spans, engine):
    eng, _, _ = engine
    assert eng.answer(QUESTION)[0] == ANSWER
    retrieval, generation, question = spans()
    assert [s.name for s in spans()] == ["rag.retrieve", "rag.generate", "rag.question"]
    assert {s.context.trace_id for s in spans()} == {question.context.trace_id}
    assert retrieval.parent.span_id == generation.parent.span_id == question.context.span_id
    assert question.parent is None
    assert all(s.attributes["rag.duration_ms"] >= 0 for s in spans())
    assert retrieval.attributes["rag.retrieval.hit.1.source"] == "fictional/policy.md"
    assert retrieval.attributes["rag.retrieval.hit.1.distance"] == 0.2
    assert generation.attributes["gen_ai.provider.name"] == "anthropic"
    assert generation.attributes["gen_ai.request.model"] == query.DEFAULT_MODEL
    assert generation.attributes["gen_ai.response.model"] == "returned-model"
    assert generation.attributes["gen_ai.usage.input_tokens"] == 12
    assert generation.attributes["gen_ai.usage.output_tokens"] == 4
    assert generation.attributes["rag.generation.fallback_used"] is True
    assert generation.attributes["rag.generation.refused"] is False
    assert question.attributes["rag.outcome"] == "generated"
    exported = json.dumps([json.loads(s.to_json()) for s in spans()])
    assert all(secret not in exported for secret in (QUESTION, DOCUMENT, ANSWER))


def test_no_match_skips_generation(spans, engine):
    eng, client, _ = engine
    eng._retrieve_vector.return_value = []
    assert eng.answer(QUESTION) == ("No relevant documents found.", [])
    assert [s.name for s in spans()] == ["rag.retrieve", "rag.question"]
    assert spans()[-1].attributes["rag.outcome"] == "no_match"
    assert spans()[-1].attributes["rag.generation.called"] is False
    client.beta.messages.create.assert_not_called()


def test_refusal_is_distinct_from_no_match(spans, engine):
    eng, _, response = engine
    response.stop_reason = "refusal"
    assert eng.answer(QUESTION)[0] == "The model declined to answer this question."
    assert spans()[1].attributes["rag.generation.refused"] is True
    assert spans()[-1].attributes["rag.outcome"] == "generated"


def test_provider_error_propagates_without_exporting_exception_text(spans, engine):
    eng, client, _ = engine
    client.beta.messages.create.side_effect = RuntimeError("PRIVATE EXCEPTION API_KEY=secret")
    with pytest.raises(RuntimeError, match="PRIVATE EXCEPTION"):
        eng.answer(QUESTION)
    for span in spans()[1:]:
        assert span.status.status_code == StatusCode.ERROR
        assert span.attributes["error.type"] == "RuntimeError"
        assert not span.events and span.status.description is None
    assert "PRIVATE" not in json.dumps([json.loads(s.to_json()) for s in spans()])


def test_deepseek_model_and_missing_usage(spans, engine, monkeypatch):
    eng, _, _ = engine
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("LLM_MODEL", "requested-model")
    client = Mock()
    client.chat.completions.create.return_value = SimpleNamespace(
        model="served-model", choices=[SimpleNamespace(
            finish_reason="stop", message=SimpleNamespace(content=ANSWER))])
    monkeypatch.setattr("openai.OpenAI", Mock(return_value=client))
    assert eng.answer(QUESTION)[0] == ANSWER
    attrs = spans()[1].attributes
    assert attrs["gen_ai.provider.name"] == "deepseek"
    assert attrs["gen_ai.request.model"] == "requested-model"
    assert attrs["gen_ai.response.model"] == "served-model"
    assert "gen_ai.usage.input_tokens" not in attrs
    assert "rag.generation.fallback_used" not in attrs


def test_unset_or_disabled_tracing_needs_no_sdk(engine, monkeypatch):
    eng, _, _ = engine
    import builtins
    original = builtins.__import__

    def without_sdk(name, *args, **kwargs):
        if name.startswith("opentelemetry"):
            raise ImportError("optional dependency missing")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_sdk)
    assert eng.answer(QUESTION)[0] == ANSWER
    query._trace_provider.cache_clear()
    monkeypatch.setenv("OTEL_TRACES_EXPORTER", "otlp")
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    assert eng.answer(QUESTION)[0] == ANSWER
    query._trace_provider.cache_clear()
    monkeypatch.delenv("OTEL_SDK_DISABLED")
    assert eng.answer(QUESTION)[0] == ANSWER


def test_missing_sdk_or_bad_config_does_not_break_answers(engine, monkeypatch, caplog):
    eng, _, _ = engine
    monkeypatch.setenv("OTEL_TRACES_EXPORTER", "unsupported-private-config")
    assert eng.answer(QUESTION)[0] == ANSWER
    assert "Tracing disabled (ValueError)" in caplog.text
    assert "unsupported-private-config" not in caplog.text


def test_span_failure_preserves_result_and_application_exception(engine, monkeypatch):
    eng, client, _ = engine
    tracer = Mock()
    tracer.start_as_current_span.side_effect = RuntimeError("PRIVATE SDK ERROR")
    monkeypatch.setattr(query, "_tracer", lambda: tracer)
    assert eng.answer(QUESTION)[0] == ANSWER
    client.beta.messages.create.side_effect = ValueError("application failure")
    with pytest.raises(ValueError, match="application failure"):
        eng.answer(QUESTION)


def test_span_end_failure_preserves_answer(engine, monkeypatch):
    eng, _, _ = engine
    tracer = Mock()
    manager = tracer.start_as_current_span.return_value
    manager.__enter__ = Mock(return_value=Mock())
    manager.__exit__ = Mock(side_effect=RuntimeError("PRIVATE SDK ERROR"))
    monkeypatch.setattr(query, "_tracer", lambda: tracer)
    assert eng.answer(QUESTION)[0] == ANSWER


def test_retrieval_error_marks_child_and_question(spans, engine):
    eng, client, _ = engine
    eng.embedder.encode.side_effect = ValueError("PRIVATE EMBEDDING FAILURE")
    with pytest.raises(ValueError, match="PRIVATE EMBEDDING FAILURE"):
        eng.answer(QUESTION)
    assert [s.name for s in spans()] == ["rag.retrieve", "rag.question"]
    assert all(s.attributes["error.type"] == "ValueError" for s in spans())
    assert all(s.status.status_code == StatusCode.ERROR for s in spans())
    client.beta.messages.create.assert_not_called()


def test_concurrent_questions_keep_separate_parents(spans, engine):
    eng, _, _ = engine
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(eng.answer, [QUESTION, QUESTION]))
        assert all(result[0] == ANSWER for result in results)
    roots = [s for s in spans() if s.name == "rag.question"]
    assert len(roots) == 2 and roots[0].context.trace_id != roots[1].context.trace_id
    for root in roots:
        children = [s for s in spans() if s.parent and s.parent.span_id == root.context.span_id]
        assert {s.name for s in children} == {"rag.retrieve", "rag.generate"}


def test_http_retrieval_has_question_parent_without_generation(spans, engine, monkeypatch):
    import api
    eng, client, _ = engine
    service = object.__new__(api.RetrievalService)
    service._lock = threading.Lock()
    service.retrieval = "vector"
    service.require_fresh = False
    service.engine = lambda: eng
    service.freshness = lambda _: {"status": "current"}
    monkeypatch.setattr(api, "corpus_identity", lambda _: {"collection": "fictional"})
    assert service.retrieve({"question": QUESTION})["hits"][0]["source"] == "fictional/policy.md"
    retrieval, question = spans()
    assert retrieval.name == "rag.retrieve" and question.name == "rag.question"
    assert retrieval.parent.span_id == question.context.span_id
    assert question.attributes["rag.generation.called"] is False
    client.beta.messages.create.assert_not_called()


def test_console_configuration_does_not_replace_host_provider(monkeypatch):
    from opentelemetry import trace
    original = trace.get_tracer_provider()
    monkeypatch.setenv("OTEL_TRACES_EXPORTER", "console")
    monkeypatch.setenv("OTEL_SERVICE_NAME", "fictional-test-service")
    provider = query._trace_provider()
    try:
        assert provider.resource.attributes["service.name"] == "fictional-test-service"
        assert trace.get_tracer_provider() is original
    finally:
        provider.shutdown()
