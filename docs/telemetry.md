# Telemetry and Jaeger

Telemetry is information about how the application ran. A **trace** follows one question;
its **spans** time the steps inside it. **OpenTelemetry (OTel)** creates and exports those spans.
**Jaeger** collects them and lets you inspect their timeline in a browser.

```text
RAG question → OpenTelemetry SDK → OTLP/HTTP :4318 → Jaeger → browser UI :16686
```

Use this to see whether a question spent its time retrieving documents or waiting for a model,
and whether it returned no matches, a model refusal, or an error. It does not measure whether
an answer is correct; use the retrieval and answer evals for that.

Run the commands below from the RAG checkout, using the same Python environment as the app.
Docker is needed only for the local Jaeger collector. A console trace needs no Docker.
The demo uses fictional data and stubbed model calls, so it needs no provider key or embedding
model download. Its durations and token counts do not benchmark a real provider.

## Run the local demo

Tracing is off by default. Set `OTEL_TRACES_EXPORTER=console` or `otlp` before starting the
CLI, UI or HTTP service and install the optional packages:

```bash
python -m pip install -r requirements-otel.txt
```

Each answered question produces this trace (durations are measured by the spans):

```text
rag.question
├── rag.retrieve   mode, k, hit count, source paths and cosine distances
└── rag.generate   provider, requested/served model, tokens, refusal/fallback
```

No-match questions have no generation span or provider call. Retrieval-only HTTP requests have
`rag.question` → `rag.retrieve`. Calling the engine's `retrieve()` helper directly produces just
the retrieval span. Errors record their class and error status; ordinary results, citations and
exceptions keep the same behavior when tracing is disabled or unavailable.

To view a fictional trace in a local Jaeger collector without a model download or paid API call:

```bash
docker run --rm -d --name rag-trace-jaeger \
  -p 127.0.0.1:16686:16686 -p 127.0.0.1:4318:4318 \
  cr.jaegertracing.io/jaegertracing/jaeger:2.22.0
export OTEL_TRACES_EXPORTER=otlp
export OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4318
export OTEL_SERVICE_NAME=rag-trace-demo
export OTEL_TRACES_SAMPLER=always_on
python trace_demo.py --verify-jaeger http://127.0.0.1:16686
```

Open <http://127.0.0.1:16686>, select **rag-trace-demo**, and find the question's three spans.
The script prints the trace ID after verifying that Jaeger received that exact trace and its
parent/child links. The demo uses deterministic retrieval and provider stubs around the production
engine methods; its token counts are fictional. CI runs this collector check too. For JSON spans
on stderr instead, run `OTEL_TRACES_EXPORTER=console python trace_demo.py`.

For real questions, keep these variables set and run `python query.py "your question"`,
`streamlit run app.py`, or `python api.py` as usual. Generated answers still require the configured
provider and its normal credentials. `otlp` uses HTTP/protobuf; standard
`OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`, headers and timeout variables are honored by the exporter.
The per-signal endpoint must include `/v1/traces`; the general endpoint above gets that path
automatically. Resource attributes and sampling use the SDK's standard OTel variables. The app
uses its own provider without replacing a host application's global provider. Configuration is
read once per process; restart after changing it. `OTEL_SDK_DISABLED=true`, `none`, or an unset
`OTEL_TRACES_EXPORTER` disables this instrumentation.

Opting in exports source paths, distances, model identifiers and usage to your configured
collector. It does not export questions, document/excerpt text, answers, credentials, or exception
messages/stacks. Source paths can themselves identify private documents: use a collector approved
for that metadata and control its access and retention. SDK attribute limits and sampling can
limit what is retained. Collector/export failures must not prevent questions from completing.

Stop the disposable collector with `docker stop rag-trace-jaeger`; its in-memory traces are lost.
See [OpenTelemetry Python exporters](https://opentelemetry.io/docs/languages/python/exporters/)
and [Jaeger's all-in-one setup](https://www.jaegertracing.io/docs/2.22/getting-started/) for deployment
configuration. Fault-diagnosis walkthroughs remain separate work in
[#36](https://github.com/redbarron23/rag-observability/issues/36).

## Run the automated scenarios

Install `requirements-dev.txt` in the active Python environment and start Docker, then run:

```bash
python telemetry_checks.py
```

The check creates its own uniquely named Jaeger container with random loopback ports and removes
it on success or failure. It does not use your running collector. No model download or paid API
call is made. Ordinary unit tests remain independent of Docker. CI runs this command and retains
`telemetry-results.json`, a sanitized summary of scenario names, pass flags and trace IDs.

| Controlled scenario | Required observation |
| --- | --- |
| Successful answer, refusal, fallback | Three linked spans; served/requested model, token counts and refusal/fallback flags match the fictional response |
| Missing usage | Token counts and fallback stay absent, not invented zero/false values |
| No match | Two spans, no generation/provider call |
| Retrieval-only HTTP request | Real authenticated HTTP request, two spans, no generation/provider call |
| Provider/retrieval error | Original exception preserved; failing child and question contain only error class/status |
| Collector stopped | Answers, retrieval and exceptions unchanged; operations and exporter waits bounded |
| Collector restarted | Same process/provider exports a fresh, distinct trace |
| Tracing unset/none, SDK disabled, sampling off | Usable answer and no OTLP application export requests |
| Console, missing optional exporter, trace endpoint override | Safe console spans; missing dependency leaves answers working; trace endpoint/protocol override wins |

A local recording proxy forwards real HTTP/protobuf exports to Jaeger. The assertions inspect
decoded OTLP payloads (including events), collected trace graphs, captured SDK logs and child
process diagnostics for sentinel question/document/answer/exception text and credentials. The
credential header is checked at the proxy but never written to the report. Source paths and
configured resource/model/usage metadata remain permitted. These checks cover controlled fixture
content; they do not prove arbitrary user-configured resource metadata is free of secrets.

Collector recovery verifies new traffic. Exporter retries are bounded and failed batches can be
dropped; the test does not promise replay of spans produced during an outage. The diagnostic
walkthrough in [#36](https://github.com/redbarron23/rag-observability/issues/36) can reuse these
controlled scenarios to explain what each result means.

## Read a trace

In Jaeger, select the service, choose a recent time range and **Find Traces**, then open the
trace. You can also open the trace ID printed by the verified demo directly at
`http://127.0.0.1:16686/trace/<trace-id>`.

| Span | What it times | What to inspect |
| --- | --- | --- |
| `rag.question` | The complete question operation | `rag.outcome`, `rag.generation.called`, error status |
| `rag.retrieve` | Embedding, search, section completion and policy-status lookup | `rag.retrieval.mode`, `k`, `hit_count`, `hit.1.source`, `hit.1.distance` |
| `rag.generate` | The provider call and response handling | `gen_ai.provider.name`, `gen_ai.request.model`, `gen_ai.response.model`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens` |

Retrieval fields in the table share the `rag.retrieval.` prefix; each returned hit has a numbered
source/distance pair. Generation also records `rag.generation.stop_reason`, `refused` and
`fallback_used` when the provider reports them. Missing usage or fallback information is omitted,
not replaced with zero or false.

A root outcome of `generated` means the model call completed; inspect the child span's refusal
flag to distinguish a refusal from an ordinary response. A `no_match` question skips generation.
An error marks the failing span and question with error status and `error.type`.

Child durations are included in the question duration: do not add the root and its children
as if they were separate work. An HTTP question can also include index loading and freshness
checks before retrieval. Direct calls to the retrieval helper have no question wrapper.

## Configuration

Set these in the environment that runs RAG, before its first question. Restart an existing
CLI, UI or API process after changing tracing configuration.

| Variable | Purpose |
| --- | --- |
| `OTEL_TRACES_EXPORTER` | `console` for JSON on stderr; `otlp` for the collector; unset or `none` disables tracing |
| `OTEL_SDK_DISABLED` | `true` disables this instrumentation even when an exporter is selected |
| `OTEL_SERVICE_NAME` | Service shown in Jaeger; defaults to `rag-observability` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | General collector base URL, e.g. `http://127.0.0.1:4318`; the SDK appends `/v1/traces` |
| `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` | Trace-specific URL, including `/v1/traces`; overrides the general endpoint |
| `OTEL_EXPORTER_OTLP_PROTOCOL`, `OTEL_EXPORTER_OTLP_TRACES_PROTOCOL` | Only `http/protobuf` is supported; the trace-specific setting takes precedence |
| `OTEL_EXPORTER_OTLP_HEADERS`, `OTEL_EXPORTER_OTLP_TRACES_HEADERS` | Collector authentication headers, interpreted by the SDK; keep credentials out of committed commands/files |
| `OTEL_EXPORTER_OTLP_TIMEOUT`, `OTEL_EXPORTER_OTLP_TRACES_TIMEOUT` | Exporter timeout settings, interpreted by the Python SDK |
| `OTEL_RESOURCE_ATTRIBUTES` | Additional SDK resource metadata exported with the spans |
| `OTEL_TRACES_SAMPLER`, `OTEL_TRACES_SAMPLER_ARG` | SDK sampling configuration; use `always_on` for the fictional demo |

Export is buffered by the SDK. The verified demo explicitly flushes before querying Jaeger.
Allow normal shutdown for other processes; force-killing a process can lose buffered spans.
Telemetry setup, span and export failures must leave the RAG result or original application
exception intact. Unsupported exporter or transport settings disable this instrumentation with a
warning until restart. Other SDK settings follow the SDK's validation and fallback behavior.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| No service or traces in Jaeger | Generate a question, check the selected service/time range, exporter variable, sampler and SDK-disabled setting; restart RAG after changing its environment |
| `Tracing disabled (ImportError)` | Install `requirements-otel.txt` in the Python environment or container running RAG |
| `Tracing disabled (ValueError)` | Choose `console`, `otlp` or `none`; OTLP transport must be `http/protobuf` |
| Export connection errors | Send OTLP to port 4318, not the UI port 16686; inspect exporter stderr and `docker logs rag-trace-jaeger` |
| Endpoint returns 404 | Check general versus trace-specific endpoint paths; Jaeger query verification uses `/api/v3/services` and `/api/v3/traces/<trace-id>` |
| No `rag.generate` span | Retrieval-only API requests and no-match results intentionally do not call a provider |
| Missing token counts or fallback flag | The provider did not report them; inspect model and stop-reason fields rather than guessing usage |
| Traces disappeared after stopping Jaeger | This disposable all-in-one collector uses in-memory storage; regenerate the demo trace |

Check collector readiness with:

```bash
curl --fail --max-time 2 http://127.0.0.1:16686/api/v3/services
```

For a console-only diagnostic without collector connectivity:

```bash
OTEL_TRACES_EXPORTER=console python trace_demo.py
```

To stop tracing, set `OTEL_TRACES_EXPORTER=none` and restart the RAG process. To stop the
sample collector, run `docker stop rag-trace-jaeger`.

## Using this with the obs-agent Compose simulation

The base obs-agent Compose simulation may pin an older RAG revision and does not forward the
OTel variables or install the optional HTTP exporter. Starting Jaeger alone will not enable
tracing inside that stack. The standalone RAG demo above is the verified setup for this change.
The opt-in combined setup is tracked in [obs-agent #111](https://github.com/redbarron23/obs-agent/issues/111).
Tracing the combined stack needs a tested RAG pin, optional packages in its image, and explicit
collector/environment configuration. A container's `127.0.0.1` refers to its own network namespace,
so it cannot use the host collector address without suitable networking.

## Data and verification boundaries

Tracing is opt-in. Source paths, model identifiers, usage and configured resource metadata go to
the chosen collector; source paths can identify private documents. Prompts, document/excerpt text,
answers, credentials and exception messages/stacks are excluded from these application spans.
Use a collector approved for the exported metadata and establish its access, storage and retention
policy before sharing it. The loopback, in-memory demo is a disposable development setup.

`tests/test_tracing.py` checks parentage, metadata, concurrent isolation, no-match/refusal/error
behavior, no-op/missing SDK operation, host-provider preservation and telemetry failures. CI also
runs the isolated collector scenarios above, verifying exact trace IDs, graph links, metadata,
privacy and recovery through the Jaeger v3 API. This validates telemetry delivery, not live model quality.

References: [OTel Python exporters](https://opentelemetry.io/docs/languages/python/exporters/),
[Jaeger setup](https://www.jaegertracing.io/docs/2.22/getting-started/), and
[Jaeger APIs](https://www.jaegertracing.io/docs/2.22/architecture/apis/).
