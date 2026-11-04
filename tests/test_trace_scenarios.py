"""Exercise the collector scenarios and their assertions without Docker/network."""

from copy import deepcopy

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import query
from telemetry_checks import private_free, verify
from trace_demo import SCENARIOS, run_scenario


@pytest.fixture
def record(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider(shutdown_on_exit=False)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(query, "_tracer", lambda: provider.get_tracer("scenario-test"))

    def read():
        result = []
        for span in exporter.get_finished_spans():
            attrs = []
            for key, value in span.attributes.items():
                kind = "boolValue" if isinstance(value, bool) else "stringValue"
                attrs.append({"key": key, "value": {kind: value}})
            result.append({"name": span.name, "spanId": f"{span.context.span_id:016x}",
                           "traceId": f"{span.context.trace_id:032x}",
                           "parentSpanId": f"{span.parent.span_id:016x}" if span.parent else "",
                           "attributes": attrs, "status": {"code": span.status.status_code.value},
                           "events": list(span.events)})
        return result

    yield read
    provider.shutdown()


@pytest.mark.parametrize("scenario", [s for s in SCENARIOS if s != "http"])
def test_fictional_scenario_and_collector_assertions(record, scenario):
    trace_id = run_scenario(scenario)
    spans = record()
    assert all(s["traceId"] == trace_id for s in spans)
    verify(spans, scenario)


@pytest.mark.parametrize("damage", ["parent", "model", "privacy", "duplicate"])
def test_collector_verification_rejects_bad_trace(record, damage):
    run_scenario()
    spans = deepcopy(record())
    if damage == "parent":
        spans[0]["parentSpanId"] = "wrong"
    elif damage == "model":
        attrs = spans[1]["attributes"]
        next(a for a in attrs if a["key"] == "gen_ai.response.model")["value"] = {"stringValue": "wrong"}
    elif damage == "privacy":
        spans[0]["events"] = [{"name": "SENTINEL PRIVATE QUESTION"}]
    else:
        spans.append(deepcopy(spans[0]))
    with pytest.raises(AssertionError):
        verify(spans, "success")


def test_diagnostics_privacy_check():
    with pytest.raises(AssertionError):
        private_free("request failed sentinel-private-credential")
