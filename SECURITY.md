# Security policy

## Supported versions

Security fixes go to `main` and ship in the next release. Only the latest release receives fixes;
there are no backport branches.

| Version | Supported |
| --- | --- |
| Latest `0.x` release | Yes |
| `main` | Yes (fixes land here first) |
| Older releases | No: upgrade to the latest release |

Releases are listed in [CHANGELOG.md](CHANGELOG.md).

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub:
**[Security → Report a vulnerability](https://github.com/redbarron23/rag-observability/security/advisories/new)**.
Don't open a public issue, discussion or pull request for a suspected vulnerability.

Include the affected version or commit, how to reproduce it, and what an attacker could do. This
is a single-maintainer project with no service-level commitment; expect an acknowledgement within
about a week. Once a fix is released, the advisory is published with credit unless you ask not
to be named.

## Trust boundaries

rag-observability is a local tool. It is safe by default only when it runs on your own machine,
over documents you trust, with a provider you are allowed to send them to. These are the
boundaries to keep in mind before changing that.

### The Streamlit UI has no authentication

- `app.py` has no login and does not read `RAG_API_KEY`. Anyone who can reach it can query every
  indexed document, upload Markdown files into `data/uploads/` and re-index, and generate answers
  that spend your provider credit.
- It listens on loopback only: [`.streamlit/config.toml`](.streamlit/config.toml) sets
  `server.address = "127.0.0.1"`. Streamlit reads that file when started from the repository
  root, as documented. Started from another directory, or with `--server.address` overridden,
  Streamlit's own default listens on all interfaces and exposes the UI to your network. Don't
  expose it beyond loopback; put it behind an authenticating reverse proxy if you must.
- Uploads must be UTF-8 `.md` files with a plain filename, at most 5 MB each. They are stored and
  indexed, never executed.

### The HTTP API has no authentication unless `RAG_API_KEY` is set

- `api.py` binds to `127.0.0.1` by default and, on loopback, rejects requests whose `Host` header
  is not a loopback name (DNS rebinding). It sends no CORS headers and requires
  `Content-Type: application/json`.
- With `RAG_API_KEY` set, every endpoint requires `Authorization: Bearer <key>`. Binding to a
  non-loopback address is refused without it.
- The server speaks plain HTTP. Put it behind TLS before traffic leaves the machine.
- There is no per-document access control: anyone who can call the API can read every indexed
  chunk. Use a separate index and server per audience.
- Details: [HTTP API access boundary](docs/http-api.md#authentication-and-access-boundary).

### Retrieved text is sent to the configured LLM provider

- Ingestion, embeddings, the vector store, retrieval, the HTTP API, retrieval evals and the
  benchmark stay local. Only the embedding model is downloaded.
- Generating an answer (`query.py`, **Generate answer** in the UI, `evals.py --with-llm`,
  `adversarial_evals.py --with-llm`) sends the question and the retrieved chunks to the configured
  provider: Anthropic, or DeepSeek at `https://api.deepseek.com`. Index only documents approved for
  that destination.
- Optional OpenTelemetry export sends span metadata to the configured OTLP endpoint. Spans exclude
  prompts, document text and answers ([telemetry privacy](docs/telemetry.md)).

### Documents can carry prompt injection

- Indexed documents are untrusted input. A document can contain instructions aimed at the model,
  false figures, or stale guidance that conflicts with current policy.
- Mitigations: chunks reach the model only as document content (a citation-enabled `document`
  block for Claude, or after `Context:` in the DeepSeek user message), never in the system prompt;
  the HTTP API marks every response `content_trust: "untrusted"` and never renders or executes
  chunk text.
- These reduce the risk but don't remove it. The
  [adversarial scenarios](docs/guide.md#adversarial-scenarios) test conflicting policies, a
  superseded runbook and two injection attacks, and document their
  [limitations](docs/guide.md#limitations). Treat answers as drafts and check the cited sections.

## In scope

- Bypassing the HTTP API's loopback, `Host` or bearer-key checks.
- Reading or writing files outside `data/` through uploads or ingestion.
- Secrets, prompts, document text or answers leaking into logs, telemetry or eval artifacts.
- A document causing code execution, or causing the HTTP API to return content as anything other
  than JSON.
- Injection techniques that defeat the mitigations above in a way the adversarial checks miss.

## Out of scope

- The Streamlit UI having no authentication, or being reachable when its loopback setting is
  bypassed (started outside the repository root, or with `--server.address` overridden).
- A model following an injected instruction in a document you chose to index, without bypassing
  the mitigations above. Please still report new techniques as ordinary issues so they can be
  added to the adversarial scenarios.
- Vulnerabilities in dependencies with no exploitable path in this project. Dependabot proposes
  dependency updates weekly.
