#!/usr/bin/env python3
"""Export a fictional question with deterministic retrieval/provider stubs; no model or API key.

Set OTEL_TRACES_EXPORTER=console or otlp. Optional --verify-jaeger polls the local query API.
"""

import argparse
import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.request import Request, urlopen

import query

QUESTION = "SENTINEL PRIVATE QUESTION"
DOCUMENT = "SENTINEL PRIVATE DOCUMENT"
ANSWER = "SENTINEL PRIVATE ANSWER"
EXCEPTION = "SENTINEL PRIVATE EXCEPTION"
CREDENTIAL = "sentinel-private-credential"
SCENARIOS = ("success", "no_match", "refusal", "fallback", "unknown_usage",
             "provider_error", "retrieval_error", "http", "slow_retrieval", "slow_generation")


def run_scenario(scenario="success"):
    assert scenario in SCENARIOS
    engine = object.__new__(query.RAGEngine)
    engine.retrieval = "vector"
    engine.embedder = Mock()
    engine.embedder.encode.return_value.tolist.return_value = [[1.0, 0.0]]
    trace_ids = []

    def fictional_retrieval(*args):
        span = query._current_trace_span.get()
        if span is not None:
            trace_ids.append(f"{span.get_span_context().trace_id:032x}")
        if scenario == "slow_retrieval":
            time.sleep(0.15)
        if scenario == "retrieval_error":
            raise ValueError(EXCEPTION)
        if scenario == "no_match":
            return []
        return [{"id": "fictional-chunk", "file": "demo-policy.md",
                 "source": "fictional/demo-policy.md", "section": "Owners",
                 "text": DOCUMENT, "distance": 0.2}]

    engine._retrieve_vector = fictional_retrieval
    engine._complete_sections = lambda embedding, chunks: chunks
    engine.policy_statuses = lambda day: None
    client = Mock()
    client.beta.messages.create.return_value = SimpleNamespace(
        model="fictional-served-model",
        stop_reason="refusal" if scenario == "refusal" else "end_turn",
        usage=(None if scenario == "unknown_usage" else SimpleNamespace(
            input_tokens=12, output_tokens=4,
            iterations=[SimpleNamespace(type="fallback_message")] if scenario == "fallback" else [])),
        content=[SimpleNamespace(type="text", text=ANSWER, citations=[])],
    )
    if scenario == "slow_generation":
        response = client.beta.messages.create.return_value

        def slow_response(**kwargs):
            time.sleep(0.15)
            return response

        client.beta.messages.create.side_effect = slow_response
    if scenario == "provider_error":
        client.beta.messages.create.side_effect = RuntimeError(EXCEPTION)
    with patch.dict("os.environ", {"LLM_PROVIDER": "anthropic", "LLM_MODEL": "fictional-demo-model"}), \
            patch("anthropic.Anthropic", return_value=client):
        if scenario == "http":
            import api
            service = object.__new__(api.RetrievalService)
            service._lock = threading.Lock()
            service.retrieval = "vector"
            service.require_fresh = False
            service.engine = lambda: engine
            service.freshness = lambda _: {"status": "current"}
            with patch.object(api, "corpus_identity", return_value={"collection": "fictional"}):
                with api.make_server(service, port=0, api_key=CREDENTIAL) as server:
                    thread = threading.Thread(target=server.serve_forever, daemon=True)
                    thread.start()
                    try:
                        request = Request(
                            f"http://127.0.0.1:{server.server_port}/retrieve",
                            data=json.dumps({"question": QUESTION}).encode(),
                            headers={"Authorization": "Bearer " + CREDENTIAL,
                                     "Content-Type": "application/json"})
                        with urlopen(request, timeout=5) as response:
                            assert json.load(response)["hits"][0]["text"] == DOCUMENT
                    finally:
                        server.shutdown()
                        thread.join(timeout=5)
        else:
            expected_error = {"provider_error": RuntimeError, "retrieval_error": ValueError}.get(scenario)
            try:
                answer, _chunks = engine.answer(QUESTION)
            except (RuntimeError, ValueError) as exc:
                assert expected_error is type(exc) and str(exc) == EXCEPTION
            else:
                assert expected_error is None, "application exception was swallowed"
                expected = {"no_match": "No relevant documents found.",
                            "refusal": "The model declined to answer this question."}.get(scenario, ANSWER)
                assert answer == expected
    if scenario in ("http", "no_match", "retrieval_error"):
        client.beta.messages.create.assert_not_called()
    return trace_ids[0] if trace_ids else None


def wait_for_trace(jaeger, trace_id, names):
    endpoint = jaeger.rstrip("/") + "/api/v3/traces/" + trace_id
    deadline = time.monotonic() + 30
    while True:
        try:
            with urlopen(endpoint, timeout=2) as response:
                result = json.load(response).get("result", {})
        except OSError:
            result = {}
        spans = [span for resource in result.get("resourceSpans", [])
                 for scope in resource.get("scopeSpans", []) for span in scope.get("spans", [])]
        if len(spans) == len(names) and {s["name"] for s in spans} == names:
            return spans
        assert time.monotonic() < deadline, "complete trace not received in 30 seconds"
        time.sleep(0.2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-jaeger", help="local Jaeger query URL, e.g. http://127.0.0.1:16686")
    parser.add_argument("--scenario", choices=SCENARIOS, default="success")
    args = parser.parse_args()
    trace_id = run_scenario(args.scenario)
    provider = query._trace_provider()
    if provider is None:
        assert not args.verify_jaeger, "Collector verification needs enabled tracing"
        print(f"PASS: {args.scenario} application assertions; tracing disabled")
        return
    assert provider.force_flush(timeout_millis=5000), "Trace export did not flush within five seconds"
    if args.verify_jaeger:
        names = {"rag.question", "rag.retrieve"}
        if args.scenario not in ("http", "no_match", "retrieval_error"):
            names.add("rag.generate")
        spans = {s["name"]: s for s in wait_for_trace(args.verify_jaeger, trace_id, names)}
        root = spans["rag.question"]
        for name, child in spans.items():
            if name == "rag.question":
                continue
            assert child.get("parentSpanId") == root["spanId"], "missing question parent"
        print(f"PASS: {args.scenario}; Jaeger received {len(spans)} linked spans; trace {trace_id}")
    else:
        print(f"PASS: {args.scenario} application assertions; trace context {trace_id}")


if __name__ == "__main__":
    main()
