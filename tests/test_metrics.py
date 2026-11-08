"""Metrics from the production engine with deterministic retrieval and mocked providers."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

import query

QUESTION = "PRIVATE QUESTION"
DOCUMENT = "PRIVATE DOCUMENT"
ANSWER = "PRIVATE ANSWER"
CHUNK_ID = "private-chunk-id#0"
SOURCE = "private/source-path.md"

# name -> (unit, attribute keys that may appear)
GENERATION_KEYS = {"gen_ai.operation.name", "gen_ai.provider.name", "gen_ai.request.model",
                   "gen_ai.response.model", "error.type"}
EXPECTED = {
    "gen_ai.client.operation.duration": ("s", GENERATION_KEYS),
    "gen_ai.client.token.usage": ("{token}", GENERATION_KEYS | {"gen_ai.token.type"}),
    "rag.retrieval.duration": ("s", {"rag.retrieval.mode", "rag.outcome", "error.type"}),
    "rag.retrieval.chunks": ("{chunk}", {"rag.retrieval.mode", "rag.chunk.disposition"}),
    "rag.questions": ("{question}", {"rag.retrieval.mode", "rag.outcome", "error.type"}),
    "rag.generation.refusals": ("{refusal}", GENERATION_KEYS),
    "rag.generation.fallbacks": ("{fallback}", GENERATION_KEYS),
    "rag.generation.errors": ("{error}", GENERATION_KEYS),
}


@pytest.fixture(autouse=True)
def metrics_off(monkeypatch):
    for key in ("OTEL_METRICS_EXPORTER", "OTEL_TRACES_EXPORTER", "OTEL_SDK_DISABLED"):
        monkeypatch.delenv(key, raising=False)
    caches = (query._meter_provider, query._instruments)
    for cached in caches:
        cached.cache_clear()
    yield
    for cached in caches:  # tests may monkeypatch the module attributes
        cached.cache_clear()


@pytest.fixture
def reader(monkeypatch):
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader], shutdown_on_exit=False)
    monkeypatch.setattr(query, "_meter_provider", lambda: provider)
    yield reader
    provider.shutdown()


def collect(reader) -> dict:
    """name -> metric for this project's instruments."""
    data = reader.get_metrics_data()
    if data is None:
        return {}
    return {m.name: m for rm in data.resource_metrics for sm in rm.scope_metrics
            for m in sm.metrics if sm.scope.name == "rag-observability"}


def points(metric) -> list:
    return list(metric.data.data_points)


def point(metric, **attrs):
    matches = [p for p in points(metric) if all(p.attributes.get(k) == v for k, v in attrs.items())]
    assert len(matches) == 1, (metric.name, attrs, [dict(p.attributes) for p in points(metric)])
    return matches[0]


def assert_schema(metrics: dict) -> None:
    for name, metric in metrics.items():
        unit, keys = EXPECTED[name]
        assert metric.unit == unit, name
        for p in points(metric):
            assert set(p.attributes) <= keys, (name, dict(p.attributes))
    exported = json.dumps([json.loads(m.to_json()) for m in metrics.values()])
    for secret in (QUESTION, DOCUMENT, ANSWER, CHUNK_ID, SOURCE, "PRIVATE"):
        assert secret not in exported


@pytest.fixture
def engine(monkeypatch):
    eng = object.__new__(query.RAGEngine)
    eng.retrieval = "vector"
    eng.embedder = Mock()
    eng.embedder.encode.return_value.tolist.return_value = [[1.0, 0.0]]
    eng._retrieve_vector = Mock(return_value=[{
        "id": CHUNK_ID, "file": "policy.md", "source": SOURCE,
        "distance": 0.2, "text": DOCUMENT, "section": "Policy",
    }])
    eng._complete_sections = lambda embedding, chunks: chunks
    eng.policy_statuses = lambda day: None
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_MODEL", "requested-model")
    response = SimpleNamespace(
        model="served-model", stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=12, output_tokens=4, iterations=[]),
        content=[SimpleNamespace(type="text", text=ANSWER, citations=[])],
    )
    client = Mock()
    client.beta.messages.create.return_value = response
    monkeypatch.setattr("anthropic.Anthropic", Mock(return_value=client))
    return eng, client, response


def test_answered_question_records_retrieval_generation_and_tokens(reader, engine):
    eng, _, _ = engine
    assert eng.answer(QUESTION)[0] == ANSWER
    metrics = collect(reader)
    assert set(metrics) == {"gen_ai.client.operation.duration", "gen_ai.client.token.usage",
                            "rag.retrieval.duration", "rag.retrieval.chunks", "rag.questions"}
    assert_schema(metrics)
    model = {"gen_ai.operation.name": "chat", "gen_ai.provider.name": "anthropic",
             "gen_ai.request.model": "requested-model", "gen_ai.response.model": "served-model"}
    duration = point(metrics["gen_ai.client.operation.duration"])
    assert dict(duration.attributes) == model and duration.count == 1 and duration.sum >= 0
    assert point(metrics["gen_ai.client.token.usage"], **{"gen_ai.token.type": "input"}).sum == 12
    output = point(metrics["gen_ai.client.token.usage"], **{"gen_ai.token.type": "output"})
    assert dict(output.attributes) == {**model, "gen_ai.token.type": "output"} and output.sum == 4
    retrieval = point(metrics["rag.retrieval.duration"])
    assert dict(retrieval.attributes) == {"rag.retrieval.mode": "vector", "rag.outcome": "matched"}
    assert retrieval.count == 1
    kept = point(metrics["rag.retrieval.chunks"])
    assert dict(kept.attributes) == {"rag.retrieval.mode": "vector", "rag.chunk.disposition": "kept"}
    assert kept.value == 1
    question = point(metrics["rag.questions"])
    assert dict(question.attributes) == {"rag.retrieval.mode": "vector", "rag.outcome": "generated"}


def test_no_match_counts_question_without_generation(reader, engine):
    eng, client, _ = engine
    eng._retrieve_vector.return_value = []
    assert eng.answer(QUESTION)[0] == "No relevant documents found."
    metrics = collect(reader)
    assert_schema(metrics)
    assert not any(name.startswith(("gen_ai.", "rag.generation.")) for name in metrics)
    assert point(metrics["rag.retrieval.duration"], **{"rag.outcome": "no_match"}).count == 1
    assert point(metrics["rag.questions"], **{"rag.outcome": "no_match"}).value == 1
    assert "rag.retrieval.chunks" not in metrics
    client.beta.messages.create.assert_not_called()


def test_refusal_and_fallback_counters(reader, engine):
    eng, _, response = engine
    response.stop_reason = "refusal"
    response.usage.iterations = [SimpleNamespace(type="fallback_message")]
    eng.answer(QUESTION)
    metrics = collect(reader)
    assert_schema(metrics)
    assert point(metrics["rag.generation.refusals"]).value == 1
    assert point(metrics["rag.generation.fallbacks"]).value == 1
    assert "rag.generation.errors" not in metrics


def test_missing_usage_records_no_token_points(reader, engine):
    eng, _, response = engine
    response.usage = None
    eng.answer(QUESTION)
    metrics = collect(reader)
    assert "gen_ai.client.token.usage" not in metrics
    assert "rag.generation.fallbacks" not in metrics
    assert point(metrics["gen_ai.client.operation.duration"]).count == 1


def test_provider_error_counts_error_class_only(reader, engine):
    eng, client, _ = engine
    client.beta.messages.create.side_effect = RuntimeError("PRIVATE EXCEPTION API_KEY=secret")
    with pytest.raises(RuntimeError, match="PRIVATE EXCEPTION"):
        eng.answer(QUESTION)
    metrics = collect(reader)
    assert_schema(metrics)
    error = point(metrics["rag.generation.errors"])
    assert error.attributes["error.type"] == "RuntimeError" and error.value == 1
    assert "gen_ai.response.model" not in error.attributes
    assert point(metrics["gen_ai.client.operation.duration"]).attributes["error.type"] == "RuntimeError"
    assert point(metrics["rag.questions"]).attributes["rag.outcome"] == "error"
    assert "gen_ai.client.token.usage" not in metrics


def test_retrieval_error_records_error_class(reader, engine):
    eng, _, _ = engine
    eng.embedder.encode.side_effect = ValueError("PRIVATE EMBEDDING FAILURE")
    with pytest.raises(ValueError):
        eng.answer(QUESTION)
    metrics = collect(reader)
    assert_schema(metrics)
    retrieval = point(metrics["rag.retrieval.duration"])
    assert dict(retrieval.attributes) == {"rag.retrieval.mode": "vector", "error.type": "ValueError"}
    assert point(metrics["rag.questions"], **{"error.type": "ValueError"}).value == 1


def test_chunk_dispositions_cover_cutoff_margin_and_section_expansion(reader, engine):
    eng, _, _ = engine
    candidates = {
        "ids": [["a#0", "b#0", "c#0"]],
        "documents": [["near text", "mentions Ops Agent here", "unrelated"]],
        "metadatas": [[{"source": "a.md"}, {"source": "b.md"}, {"source": "c.md"}]],
        "distances": [[0.3, 0.65, 0.9]],
    }
    eng.collection = Mock()
    eng.collection.query.return_value = candidates
    del eng._retrieve_vector  # use the real ranking and cutoff
    sibling = {"id": "a#1", "text": "sibling", "source": "a.md", "file": "", "section": "S",
               "distance": 0.8, "expanded_from": "a#0"}
    eng._complete_sections = lambda embedding, chunks: [*chunks, sibling]
    eng.retrieve("What does the Ops Agent do?", n_results=3)
    metrics = collect(reader)
    assert_schema(metrics)
    chunks = metrics["rag.retrieval.chunks"]
    counts = {p.attributes["rag.chunk.disposition"]: p.value for p in points(chunks)}
    assert counts == {"kept": 1, "kept_exact_term": 1, "section_expanded": 1, "dropped": 1}


def test_hybrid_counts_drops_among_top_k_only(reader, engine):
    eng, _, _ = engine
    eng.retrieval = "hybrid"
    eng._corpus = {
        "ids": ["a#0", "b#0", "c#0"], "documents": ["alpha", "beta", "gamma"],
        "metadatas": [{"source": "a.md"}, {"source": "b.md"}, {"source": "c.md"}],
        "embeddings": [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], "tokens": [["alpha"], ["beta"], ["gamma"]],
    }
    eng.retrieve("alpha", n_results=2)
    counts = {p.attributes["rag.chunk.disposition"]: p.value
              for p in points(collect(reader)["rag.retrieval.chunks"])}
    assert counts == {"kept": 1, "dropped": 1}


@pytest.mark.parametrize("env", [{}, {"OTEL_METRICS_EXPORTER": "none"},
                                 {"OTEL_METRICS_EXPORTER": "console", "OTEL_SDK_DISABLED": "true"},
                                 {"OTEL_TRACES_EXPORTER": "console"}])
def test_metrics_off_by_default_emit_nothing(engine, monkeypatch, env):
    eng, _, _ = engine
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    created = []
    original = MeterProvider.__init__

    def spy(self, *args, **kwargs):
        created.append(self)
        original(self, *args, **kwargs)

    monkeypatch.setattr(MeterProvider, "__init__", spy)
    assert eng.answer(QUESTION)[0] == ANSWER
    assert query._meter_provider() is None and query._instruments() is None
    assert not created


def test_bad_metrics_config_does_not_break_answers(engine, monkeypatch, caplog):
    eng, _, _ = engine
    monkeypatch.setenv("OTEL_METRICS_EXPORTER", "unsupported-private-config")
    assert eng.answer(QUESTION)[0] == ANSWER
    assert "Metrics disabled (ValueError)" in caplog.text
    assert "unsupported-private-config" not in caplog.text


def test_grpc_protocol_is_rejected(monkeypatch, caplog):
    monkeypatch.setenv("OTEL_METRICS_EXPORTER", "otlp")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_METRICS_PROTOCOL", "grpc")
    assert query._meter_provider() is None
    assert "Metrics disabled (ValueError)" in caplog.text


def test_recording_failure_preserves_answer(engine, monkeypatch):
    eng, _, _ = engine
    broken = Mock()
    broken.record.side_effect = RuntimeError("PRIVATE SDK ERROR")
    broken.add.side_effect = RuntimeError("PRIVATE SDK ERROR")
    monkeypatch.setattr(query, "_instruments", lambda: dict.fromkeys(
        ("operation_duration", "token_usage", "retrieval_duration", "chunks", "questions",
         "refusals", "fallbacks", "errors"), broken))
    assert eng.answer(QUESTION)[0] == ANSWER


def test_console_configuration_does_not_replace_host_provider(monkeypatch):
    from opentelemetry import metrics
    original = metrics.get_meter_provider()
    monkeypatch.setenv("OTEL_METRICS_EXPORTER", "console")
    monkeypatch.setenv("OTEL_SERVICE_NAME", "fictional-test-service")
    provider = query._meter_provider()
    try:
        assert provider is not None
        assert provider._sdk_config.resource.attributes["service.name"] == "fictional-test-service"
        assert metrics.get_meter_provider() is original
        assert set(query._instruments()) == {
            "operation_duration", "token_usage", "retrieval_duration", "chunks", "questions",
            "refusals", "fallbacks", "errors"}
    finally:
        provider.shutdown()
