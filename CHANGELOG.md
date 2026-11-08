# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Releases are Git tags named `vX.Y.Z`
with matching [GitHub releases](https://github.com/redbarron23/rag-observability/releases).

## Versioning

There are two version numbers, and they change independently:

| Version | Where | Covers |
| --- | --- | --- |
| **Project version** (this file, Git tags) | `vX.Y.Z` tags and the headings below | The whole repo: CLI commands and flags, environment variables, the on-disk index, the eval run artifact, the Streamlit app and the HTTP API implementation |
| **Contract version** | `CONTRACT_VERSION` in `api.py`, `x-contract-version` in [`contract/v1/`](contract/v1/) | Only the HTTP wire format of `POST /retrieve` and `GET /health`; rules in [docs/http-api.md](docs/http-api.md#version-numbers) |

- A release states the contract version it serves. Most releases leave the contract unchanged.
- A contract MINOR bump (an additive wire change) is at least a project MINOR bump.
- A contract MAJOR bump (a breaking wire change) is a breaking project release: a MINOR bump while
  the project is `0.x`, a MAJOR bump from `1.0.0` on.
- While the project is `0.x`, a MINOR release may change CLI flags, defaults or the index layout;
  the entry says so under **Changed** or **Removed**. PATCH releases only fix bugs or documentation.
- Retrieval results (which chunks rank where) are covered by neither version; the retrieval evals
  guard them.

Consumers such as obs-agent should pin a release tag for code and send
`X-RAG-Contract-Version` for the wire format. See
[docs/http-api.md](docs/http-api.md#pinning-and-upgrading).

## [Unreleased]

## [0.1.0] - 2026-11-08

First tagged release. It serves retrieval API **contract 1.1.0**. Everything below was merged
between 4 October and 8 November 2026 without a release; PR numbers are given for reference.

### Added

- **Pipeline.** Heading-aware Markdown chunking, local `all-MiniLM-L6-v2` embeddings in ChromaDB,
  and cited answers from Claude (default `claude-sonnet-5-5`) or DeepSeek through `query.py`
  (#2, #3).
- **Retrieval.** A default cosine-distance cutoff of 0.6 (#3, #26), an exact-term margin that
  keeps near-cutoff chunks containing a term from the question (#63, #80), completion of sections
  split across chunks (#75), and opt-in hybrid BM25 + vector retrieval with `--retrieval hybrid`
  (#55).
- **Native citations.** On the Anthropic path each chunk is sent as a document with citations
  enabled, and the CLI lists the quoted text (#53).
- **Ingestion.** Incremental ingest that re-embeds only changed files (#54), a rebuild that keeps
  the working index when it fails (#51), corpus provenance with `ingest.py --check` for stale
  sources (#52), recorded embedding versions with automatic re-embed on change, and
  `ingest.py --prune` for superseded collections (#70).
- **Policy metadata.** Optional front matter (`policy_id`, `version`, effective dates,
  `supersedes`) carried through retrieval as a per-hit applicability status (#84).
- **Streamlit app.** A local document explorer with source inspection, uploads, a retrieval-mode
  selector and quoted citations (#41, #68), plus fictional upload samples and a manual test plan
  (#57).
- **Retrieval HTTP API.** `api.py` serves `POST /retrieve` and `GET /health` for obs-agent,
  retrieval-only, loopback by default with optional bearer auth via `RAG_API_KEY` (#67). A
  versioned JSON Schema contract with replayed examples in `contract/v1/` (contract 1.0.0, #81;
  1.1.0 adds `policy` on each hit, #84).
- **Evals.** Retrieval evals for sources, supporting sections and fact evidence that run in CI
  without a key; opt-in `--with-llm` answer evals for facts, fact pairs, citations and declines on
  unanswerable questions; eight held-out questions (#26, #41, #73, #75). Reproducible JSON run
  artifacts with `--output` (#77). A weekly and manual LLM eval workflow that never runs on pull
  requests (#69).
- **Adversarial checks.** `adversarial_evals.py` covers conflicting, superseded and injected
  documents over a separate fictional corpus (#82), with structural recognition of reported
  injections (#101).
- **Telemetry.** Optional OpenTelemetry spans for questions, retrieval and generation that exclude
  prompts, document text and answers (#85), real-collector scenario, privacy and recovery checks
  in CI (#92), a hands-on diagnosis lab (#93), and W3C trace context on authenticated API requests
  (#94, #95).
- **Benchmark.** `benchmark.py` compares BM25 keyword, vector and hybrid retrieval with latency
  (#102).
- **Corpus.** Fictional sample documents, including a tagging standard aligned with the obs-agent
  vocabulary (#66), a logging-cost spike runbook (#74), vendor source links for Azure metrics
  (#79) and measurable coverage and maturity criteria (#96).
- **Tooling.** Ruff linting in CI (#16) and Dependabot for pip and GitHub Actions (#15).

### Changed

- Default retrieval depth raised to 6 chunks (#2).
- README shortened, with details moved to `docs/guide.md` (#1, #17).
- Dependencies pinned below their next major (minor for Ruff) and bumped through Dependabot:
  chromadb 1.5.9, sentence-transformers 6.1, anthropic 1.13, openai 3.42, pytest 9.2,
  jsonschema 4.26, opentelemetry-sdk 1.45.2, ruff 0.17.3, and GitHub Actions checkout v7,
  setup-python v7 and cache v6 (#15, #18–#25, #87–#91).
- LLM eval cost documented as the measured $0.42 per retrieval mode per run (#103).

### Fixed

- Eval error accounting: provider failures count as failures, never as passes, and Markdown
  chunking edge cases (#40).
- Prompt, README drift and eval checks (#3); the Tier 1 README example (#1).
- Azure metrics query guidance in the sample corpus (#79).

### Security

- The HTTP API binds to loopback, rejects non-loopback `Host` headers, refuses non-loopback binds
  without `RAG_API_KEY`, and marks every response `content_trust: "untrusted"` (#67, #81).
- Telemetry excludes prompts, document text, answers, credentials and baggage (#85, #92, #94).

[Unreleased]: https://github.com/redbarron23/rag-observability/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/redbarron23/rag-observability/releases/tag/v0.1.0
