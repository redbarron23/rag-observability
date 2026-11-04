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

## Hands-on diagnosis lab

**Predict first, run second, explain third.** A trace is a timeline for one operation. A span is
one timed step; a parent contains its children. Attributes are named facts on those spans.
Sampling decides which operations are recorded. OTel makes/exports the records; Jaeger stores
and displays them. The application can work even when no record reaches Jaeger.

Use the optional dependencies and disposable Jaeger setup above. The fixed task is to answer the
fictional fixture question `SENTINEL PRIVATE QUESTION` from one fictional owner-policy chunk.
The model, embedding and document content are stubs. These are controlled experiments, not a
quality or performance benchmark. Before each command, predict the span graph and the result.

```bash
export OTEL_TRACES_EXPORTER=otlp
export OTEL_TRACES_SAMPLER=always_on
export OTEL_SERVICE_NAME=rag-trace-demo
export OTEL_EXPORTER_OTLP_ENDPOINT=http://127.0.0.1:4318
# Clear conflicting trace-specific endpoints/headers or use a clean shell for the demo.
python trace_demo.py --scenario success --verify-jaeger http://127.0.0.1:16686
```

For each row below run the same command, replacing `success` with the scenario name. The script
asserts the expected application answer, miss, refusal or original error before flushing. Verification
requires the exact trace ID and expected parent links. `telemetry_checks.py` additionally checks
all metadata, privacy and fault assertions; its final verdict is `PASS: 19 telemetry scenarios`.

| Scenario / task | Controlled change and expectation | Evidence and next diagnostic action | Feynman answer key |
| --- | --- | --- | --- |
| `success`: establish the comparison | One retrieved chunk and normal fictional answer; question with retrieve/generate children; test passes | `hit_count=1`, outcome `generated`, 12 input/4 output tokens; inspect the source before assessing correctness | “We found a document and the model answered. The timeline proves the steps ran, not that the answer is right.” |
| `slow_retrieval`: locate retrieval delay | Add 150 ms inside retrieval, no delay in provider stub; three spans; test passes | Retrieve `rag.duration_ms >= 150`; compare its width with generation; investigate embedding/search/index load in a real trace | “Finding the document took the extra time. I would check the search step first.” |
| `slow_generation`: locate provider delay | Add 150 ms only in the provider response; three spans; test passes | Generate duration at least 150 ms; inspect provider latency/retries/token volume next | “The document was found, then the answer step waited. I would check the provider.” |
| `no_match`: explain missing generation | Return no retrieval hits; two spans, no provider call; test passes | Question outcome `no_match`, generation.called false, hit_count 0; inspect corpus, cutoff and wording | “There was nothing suitable to answer from, so the model was never called.” |
| `provider_error`: locate a failed model call | Stub raises RuntimeError; three spans; original exception expected, so test passes | Question and generate have ERROR/error.type RuntimeError, no exception message/event; check provider availability/configuration | “Search worked, but the answer step failed. The test passed because it expected that failure and checked the evidence.” |
| `retrieval_error`: locate failed search | Retrieval stub raises ValueError; question/retrieve only; test passes | Both spans ERROR/error.type ValueError; no generation; inspect index/embedding inputs next | “Search failed before there was evidence to send to a model.” |
| `refusal`: distinguish decline from missing evidence | Provider reports stop_reason refusal; three spans and generated root outcome; test passes | Generate refused true; inspect stop reason rather than treating generated as a useful answer | “We found evidence and called the model, but it declined.” |
| `fallback`: inspect reported routing | Stub reports a fallback iteration; three spans; test passes | fallback_used true, requested fictional-demo-model versus served fictional-served-model; inspect routing policy next | “The provider says another model served the answer. That is reported routing, not a guess from latency.” |
| `unknown_usage`: avoid invented facts | Response omits usage; three spans; test passes | Token counts and fallback absent; inspect provider reporting, do not replace them with zero/false | “We do not know the token use or fallback state from this response.” |
| `http`: separate retrieval from answering | Real authenticated retrieval-only HTTP request; two spans; test passes | generation.called false, hit_count 1; inspect API contract before expecting a model span | “This caller asked for documents, so no model answer was requested.” |

In Jaeger select **rag-trace-demo**, a recent time range and **Find Traces**, or open the printed
trace ID at `/trace/<trace-id>`. Expand the question, compare its children, then inspect the named
attributes in the table. Parent duration already includes child duration: do not sum all three.
A wide question with short children can include setup or other work outside the child steps.

For **every row**, explain aloud in three sentences: “I predicted __. The trace/result shows __
because the fields __ say so. I would check __ next.” Use the final column to check your reasoning.
For the two slow cases predict which child widens *before* running; for either exception predict
whether generation will exist. Missing spans can reflect sampling or delivery problems too.

### Outage, recovery and invisible traces

Run the complete self-cleaning exercise:

```bash
python telemetry_checks.py
```

It creates a uniquely named collector on random loopback ports, uses the same scenario stubs,
stops only that collector, then restores it. During the outage, successful answers, no-match,
HTTP retrieval and original exceptions must stay unchanged, each operation under five seconds.
Exporter waits are bounded separately. A passing `collector_outage` entry means those assertions
held; it does not mean delivery succeeded. `same_process_recovery` requires a fresh collected trace
from the original provider instance. Failed batches may be dropped; do not expect replay.
Cleanup removes only the collector created by this command, including on failure.
For the Feynman exercise, predict whether stopping Jaeger should break an answer, then explain
why the two pass flags differ from a successful export. Answer key: RAG remains usable while
telemetry delivery fails; a new collected trace after restart proves recovery, not replay.

The same command starts fresh processes for unset/none/SDK-disabled tracing and sampling-off.
Those must return a usable answer with **zero new application export requests**. Console output
must contain spans without private fixture text; a missing optional exporter must leave the answer
usable. Trace-specific endpoint/protocol settings must override the general settings.

To try invisible traces yourself, use either command below after predicting the result:

```bash
OTEL_TRACES_EXPORTER=none python trace_demo.py --scenario success
OTEL_TRACES_EXPORTER=console OTEL_TRACES_SAMPLER=always_off python trace_demo.py --scenario success
```

Do not add `--verify-jaeger` to disabled/sampled-out runs. No exported trace is expected; the
application assertions still pass. For the Feynman exercise, explain why “no trace” alone does
not prove the app failed. Answer: recording/delivery can be off while the app works; check
exporter, sampling, process restart and endpoint settings before changing retrieval.

### Deliberately misleading result

Generate `http`, then look for `rag.generate`. It is absent. Predict two possible causes before
reading on: the request was retrieval-only, or generation was skipped for a retrieval miss.
Discriminate with hit_count (1 here versus 0), root outcome (matched versus no_match), and the
actual API task. An incomplete export is another possibility if the expected graph is not complete.
Also try the wrong Jaeger service or an old time range: the browser will show nothing even when
exact-ID verification passed. Use the printed ID, then fix the filter; do not diagnose a RAG outage
from an empty search screen. Explain which observation ruled out each alternative.

### Read the test verdict and retained results

There are three separate questions: **what did RAG do**, **what telemetry arrived**, and **did that
match the test's expectation**? An intentionally injected provider exception can be an application
failure, successful error instrumentation, and a passing test at the same time. A normal answer
with missing required trace metadata should fail the telemetry test. Neither verdict establishes
answer correctness; retrieval and generation evals cover different evidence.

`telemetry-results.json` retains sanitized span graphs, attributes and measured durations for
collected scenarios, plus Python/SDK/exporter versions, source commit and retrieval mode. It records
pass flags and export counts for configuration/outage cases. CI uploads it as **telemetry-results**
for seven days, including partial progress on failure. No question, document, answer, credential or
exception text belongs in that artifact. Source paths and configured model/resource metadata are
permitted. Durations are measured around deterministic in-process stubs, with a 150 ms sleep only
in the named slow step. The first request initializes tracing; later requests reuse the provider.
There is no real embedding-model cold/warm benchmark or hosted-provider latency measurement here.

See the representative captured run below for actual observations beside these expectations.
To finish a manual Jaeger session, run `docker stop rag-trace-jaeger`; in-memory traces disappear.
Unset the demo's OTel variables or close its shell before using a different data context.

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
