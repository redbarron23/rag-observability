#!/usr/bin/env python3
"""Disposable real-collector checks; no embedding downloads or paid provider calls."""

import io
import json
import logging
import os
import platform
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.metadata import version
from pathlib import Path
from urllib.request import Request, urlopen
from uuid import uuid4

from google.protobuf.json_format import MessageToDict
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

import query
from trace_demo import (
    ANSWER,
    CREDENTIAL,
    DOCUMENT,
    EXCEPTION,
    QUESTION,
    SCENARIOS,
    run_scenario,
    wait_for_trace,
)

SECRETS = (QUESTION, DOCUMENT, ANSWER, EXCEPTION, CREDENTIAL)


def private_free(value):
    assert all(secret not in value for secret in SECRETS), "private sentinel in telemetry/diagnostics"


def attributes(span):
    return {a["key"]: next(iter(a["value"].values())) for a in span.get("attributes", [])}


def names_for(scenario):
    names = {"rag.question", "rag.retrieve"}
    if scenario not in ("http", "no_match", "retrieval_error"):
        names.add("rag.generate")
    return names


def verify(spans, scenario):
    private_free(json.dumps(spans))
    named = {s["name"]: s for s in spans}
    assert len(named) == len(spans) and set(named) == names_for(scenario)
    root = named["rag.question"]
    assert not root.get("parentSpanId"), "question must be a root"
    assert len({s["traceId"] for s in spans}) == 1
    for name, child in named.items():
        if name != "rag.question":
            assert child["parentSpanId"] == root["spanId"]
        assert float(attributes(child)["rag.duration_ms"]) >= 0
    retrieval = attributes(named["rag.retrieve"])
    assert retrieval["rag.retrieval.mode"] == "vector"
    if scenario != "retrieval_error":
        assert int(retrieval["rag.retrieval.hit_count"]) == (0 if scenario == "no_match" else 1)
        if scenario != "no_match":
            assert retrieval["rag.retrieval.hit.1.source"] == "fictional/demo-policy.md"
            assert float(retrieval["rag.retrieval.hit.1.distance"]) == 0.2
    if scenario in ("slow_retrieval", "slow_generation"):
        slow = "rag.retrieve" if scenario == "slow_retrieval" else "rag.generate"
        assert float(attributes(named[slow])["rag.duration_ms"]) >= 150
    error = {"provider_error": "RuntimeError", "retrieval_error": "ValueError"}.get(scenario)
    if error:
        failing = "rag.generate" if scenario == "provider_error" else "rag.retrieve"
        for name in ("rag.question", failing):
            assert attributes(named[name])["error.type"] == error
            assert named[name]["status"]["code"] in (2, "STATUS_CODE_ERROR")
            assert not named[name].get("events")
    elif scenario in ("http", "no_match"):
        assert attributes(root)["rag.generation.called"] is False
    if "rag.generate" in named:
        gen = attributes(named["rag.generate"])
        assert gen["gen_ai.provider.name"] == "anthropic"
        assert gen["gen_ai.request.model"] == "fictional-demo-model"
        if scenario == "provider_error":
            for key in ("gen_ai.response.model", "gen_ai.usage.input_tokens",
                        "rag.generation.refused", "rag.generation.fallback_used"):
                assert key not in gen
        else:
            assert gen["gen_ai.response.model"] == "fictional-served-model"
            assert gen["rag.generation.refused"] is (scenario == "refusal")
            if scenario == "unknown_usage":
                assert "gen_ai.usage.input_tokens" not in gen
                assert "gen_ai.usage.output_tokens" not in gen
                assert "rag.generation.fallback_used" not in gen
            else:
                assert int(gen["gen_ai.usage.input_tokens"]) == 12
                assert int(gen["gen_ai.usage.output_tokens"]) == 4
                assert gen["rag.generation.fallback_used"] is (scenario == "fallback")


def docker(*args):
    return subprocess.run(["docker", *args], check=True, capture_output=True,
                          text=True, timeout=60).stdout.strip()


def check():
    original_env = os.environ.copy()
    name = "rag-telemetry-" + uuid4().hex[:12]
    payloads, reports = [], []
    diagnostics = io.StringIO()
    handler = logging.StreamHandler(diagnostics)
    logging.getLogger().addHandler(handler)
    provider = proxy = None
    completed = False
    target = [None]

    class RecordingProxy(BaseHTTPRequestHandler):
        def do_POST(self):
            assert self.path == "/v1/traces"
            assert self.headers.get("Authorization") == "Bearer " + CREDENTIAL
            body = self.rfile.read(int(self.headers["Content-Length"]))
            decoded = MessageToDict(ExportTraceServiceRequest.FromString(body))
            private_free(json.dumps(decoded))
            payloads.append(decoded)
            try:
                request = Request(target[0], data=body,
                                  headers={"Content-Type": "application/x-protobuf"})
                with urlopen(request, timeout=1) as response:
                    response.read()
                self.send_response(200)
            except OSError:
                self.send_response(503)
            self.end_headers()

        def log_message(self, *_):
            pass

    def collector_ready():
        # Docker may assign new host ports when a stopped container restarts.
        port = docker("port", name, "4318/tcp").rsplit(":", 1)[1]
        target[0] = f"http://127.0.0.1:{port}/v1/traces"
        query_port = docker("port", name, "16686/tcp").rsplit(":", 1)[1]
        jaeger = f"http://127.0.0.1:{query_port}"
        deadline = time.monotonic() + 30
        while True:
            try:
                with urlopen(jaeger + "/api/v3/services", timeout=1):
                    return jaeger
            except OSError:
                assert time.monotonic() < deadline, "collector not ready"
                time.sleep(0.2)

    try:
        docker("run", "-d", "--name", name, "-p", "127.0.0.1::4318", "-p", "127.0.0.1::16686",
               "cr.jaegertracing.io/jaegertracing/jaeger:2.22.0")
        jaeger = collector_ready()
        proxy = ThreadingHTTPServer(("127.0.0.1", 0), RecordingProxy)
        thread = threading.Thread(target=proxy.serve_forever, daemon=True)
        thread.start()
        # Ignore ambient tracing settings: checks must never export to a user's collector.
        for key in list(os.environ):
            if key.startswith("OTEL_"):
                del os.environ[key]
        os.environ.update({
            "OTEL_TRACES_EXPORTER": "otlp", "OTEL_TRACES_SAMPLER": "always_on",
            "OTEL_SERVICE_NAME": name,
            "OTEL_EXPORTER_OTLP_ENDPOINT": f"http://127.0.0.1:{proxy.server_port}",
            "OTEL_EXPORTER_OTLP_TRACES_HEADERS": "Authorization=Bearer%20" + CREDENTIAL,
            "OTEL_EXPORTER_OTLP_TRACES_TIMEOUT": "1",
            "OTEL_BSP_EXPORT_TIMEOUT": "2000",
        })
        query._trace_provider.cache_clear()
        seen = set()

        def collected(scenario):
            nonlocal provider
            trace_id = run_scenario(scenario)
            provider = query._trace_provider()
            assert trace_id and trace_id not in seen
            seen.add(trace_id)
            assert query._trace_provider().force_flush(timeout_millis=5000)
            spans = wait_for_trace(jaeger, trace_id, names_for(scenario))
            verify(spans, scenario)
            reports.append({"scenario": scenario, "trace_id": trace_id, "passed": True,
                            "spans": spans})

        for scenario in SCENARIOS:
            collected(scenario)
        provider = query._trace_provider()
        docker("stop", "-t", "1", name)
        for scenario in ("success", "no_match", "provider_error", "retrieval_error", "http"):
            started = time.monotonic()
            run_scenario(scenario)
            assert time.monotonic() - started < 5, "exporter delayed application operation"
        provider.force_flush(timeout_millis=3000)
        reports.append({"scenario": "collector_outage", "passed": True})
        docker("start", name)
        jaeger = collector_ready()
        assert query._trace_provider() is provider
        collected("success")
        reports[-1]["scenario"] = "same_process_recovery"

        # Each configuration starts a fresh SDK, avoiding cached providers/global state.
        with tempfile.TemporaryDirectory() as temp:
            for variant in ("unset", "none", "sdk_disabled", "sampling_off", "console",
                            "missing_exporter", "trace_endpoint"):
                env = os.environ.copy()
                if variant == "unset":
                    env.pop("OTEL_TRACES_EXPORTER")
                elif variant in ("none", "console"):
                    env["OTEL_TRACES_EXPORTER"] = variant
                elif variant == "sdk_disabled":
                    env["OTEL_SDK_DISABLED"] = "true"
                elif variant == "sampling_off":
                    env["OTEL_TRACES_SAMPLER"] = "always_off"
                elif variant == "trace_endpoint":
                    env["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] = env["OTEL_EXPORTER_OTLP_ENDPOINT"] + "/v1/traces"
                    env["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://127.0.0.1:1"
                    env["OTEL_EXPORTER_OTLP_PROTOCOL"] = "grpc"
                    env["OTEL_EXPORTER_OTLP_TRACES_PROTOCOL"] = "http/protobuf"
                report = str(Path(temp) / "result.json")
                code = '''import json, sys
import query
from trace_demo import run_scenario
trace_id = run_scenario()
p = query._trace_provider()
if p is not None:
    assert p.force_flush(timeout_millis=5000)
    p.shutdown()
with open(sys.argv[1], "w") as f:
    json.dump({"trace_id": trace_id, "provider": p is not None}, f)
'''
                if variant == "missing_exporter":
                    code = '''import builtins
original_import = builtins.__import__
def missing(name, *args, **kwargs):
    if name.startswith("opentelemetry.exporter.otlp.proto.http"):
        raise ImportError("optional HTTP exporter missing")
    return original_import(name, *args, **kwargs)
builtins.__import__ = missing
''' + code
                before = len(payloads)
                result = subprocess.run([sys.executable, "-c", code, report], env=env,
                                        capture_output=True, text=True, timeout=60)
                private_free(result.stdout + result.stderr)
                assert result.returncode == 0, f"configuration failed: {variant}"
                saved = json.loads(Path(report).read_text())
                if variant == "trace_endpoint":
                    assert len(payloads) > before
                    verify(wait_for_trace(jaeger, saved["trace_id"], names_for("success")), "success")
                else:
                    assert len(payloads) == before, f"unexpected export: {variant}"
                if variant == "console":
                    assert '"rag.question"' in result.stderr and '"rag.generate"' in result.stderr
                if variant in ("unset", "none", "sdk_disabled", "missing_exporter"):
                    assert saved["provider"] is False
                reports.append({"scenario": variant, "passed": True})
        private_free(json.dumps(payloads))
        private_free(diagnostics.getvalue())
        completed = True
        print(f"PASS: {len(reports)} telemetry scenarios; sanitized telemetry-results.json")
    finally:
        if provider is not None:
            provider.shutdown()
        if proxy is not None:
            proxy.shutdown()
            proxy.server_close()
            thread.join(timeout=5)
        Path("telemetry-results.json").write_text(json.dumps(
            {"completed": completed, "scenarios": reports,
             "environment": {"commit": subprocess.run(
                 ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip(),
                 "python": platform.python_version(), "sdk": version("opentelemetry-sdk"),
                 "exporter": version("opentelemetry-exporter-otlp-proto-http"),
                 "retrieval": "vector", "timing": "measured deterministic stubs; 150ms injected delay"},
             "export_requests": len(payloads), "diagnostic_lines": len(diagnostics.getvalue().splitlines())},
            indent=2) + "\n")
        logging.getLogger().removeHandler(handler)
        os.environ.clear()
        os.environ.update(original_env)
        # Remove only the container owned by this run, even after a failed check.
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=30)


if __name__ == "__main__":
    check()
