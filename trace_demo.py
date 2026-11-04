#!/usr/bin/env python3
"""Export a fictional question with deterministic retrieval/provider stubs; no model or API key.

Set OTEL_TRACES_EXPORTER=console or otlp. Optional --verify-jaeger polls the local query API.
"""

import argparse
import json
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.error import URLError
from urllib.request import urlopen

import query


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-jaeger", help="local Jaeger query URL, e.g. http://127.0.0.1:16686")
    args = parser.parse_args()
    engine = object.__new__(query.RAGEngine)
    engine.retrieval = "vector"
    engine.embedder = Mock()
    engine.embedder.encode.return_value.tolist.return_value = [[1.0, 0.0]]
    trace_ids = []

    def fictional_retrieval(*args):
        span = query._current_trace_span.get()
        if span is not None:
            trace_ids.append(f"{span.get_span_context().trace_id:032x}")
        return [{"source": "fictional/demo-policy.md", "section": "Owners",
                 "text": "Every resource needs an owner.", "distance": 0.2}]

    engine._retrieve_vector = fictional_retrieval
    engine._complete_sections = lambda embedding, chunks: chunks
    engine.policy_statuses = lambda day: None
    client = Mock()
    client.beta.messages.create.return_value = SimpleNamespace(
        model="fictional-demo-model", stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=12, output_tokens=4, iterations=[]),
        content=[SimpleNamespace(type="text", text="Every resource needs an owner.", citations=[])],
    )
    with patch.dict("os.environ", {"LLM_PROVIDER": "anthropic", "LLM_MODEL": "fictional-demo-model"}), \
            patch("anthropic.Anthropic", return_value=client):
        engine.answer("What tag does a fictional resource need?")
    provider = query._trace_provider()
    assert provider is not None, "Set OTEL_TRACES_EXPORTER=console or otlp and install requirements-otel.txt"
    assert provider.force_flush(timeout_millis=5000), "Trace export did not flush within five seconds"
    if args.verify_jaeger:
        endpoint = args.verify_jaeger.rstrip("/") + "/api/v3/traces/" + trace_ids[0]
        deadline = time.monotonic() + 30
        while True:
            try:
                with urlopen(endpoint, timeout=2) as response:
                    result = json.load(response).get("result", {})
            except (URLError, TimeoutError):
                result = {}
            spans = {span["name"]: span for resource in result.get("resourceSpans", [])
                     for scope in resource.get("scopeSpans", []) for span in scope.get("spans", [])}
            if set(spans) == {"rag.question", "rag.retrieve", "rag.generate"}:
                root = spans["rag.question"]
                for child in (spans["rag.retrieve"], spans["rag.generate"]):
                    assert child.get("parentSpanId") == root["spanId"], "missing question parent"
                print(f"PASS: Jaeger received question/retrieve/generate trace {trace_ids[0]}")
                return
            assert time.monotonic() < deadline, "Jaeger did not return the three-span trace in 30 seconds"
            time.sleep(0.2)


if __name__ == "__main__":
    main()
