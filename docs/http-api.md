# Retrieval HTTP API

`api.py` serves the existing retrieval engine (`RAGEngine.retrieve()`) over local HTTP so other
tools (for example obs-agent) can fetch supporting chunks without the CLI or UI.

It is **retrieval-only**: it never calls Claude, DeepSeek or any other LLM, needs no provider key,
and returns no generated answer. The caller generates and cites its own answer from the returned
chunks. It uses only the Python standard library on top of `requirements.txt`.

The wire format is a versioned contract, currently **1.0.0**, published as plain JSON Schema with
example requests and responses in [`contract/v1/`](../contract/v1/). See
[Contract and versioning](#contract-and-versioning) for how a client pins it and what may change.

## Start the server

```bash
python ingest.py            # build or update the index
python api.py               # http://127.0.0.1:8765
```

| Flag | Purpose | Default |
| --- | --- | --- |
| `--host` | Bind address. Anything other than loopback requires `RAG_API_KEY` | `127.0.0.1` |
| `--port` | Port | `8765` |
| `--persist-dir` | Chroma index directory | `chroma_db` |
| `--data-dir` | Source directory to compare for freshness | the directory that was ingested |
| `--retrieval` | `vector` or `hybrid` (BM25 + vector), same as `query.py` | `vector` |
| `--require-fresh` | Return `503 index_stale` instead of results when sources changed since ingestion | off |

The embedding model loads once at startup and is reused for every request and across index reloads.

## Endpoints

### `POST /retrieve`

Request (`Content-Type: application/json`, body at most 16 KiB; unknown fields are rejected;
schema: [`retrieve-request.schema.json`](../contract/v1/schemas/retrieve-request.schema.json)):

| Field | Type | Rules | Default |
| --- | --- | --- | --- |
| `question` | string | 1–2000 characters after trimming; no NUL | required |
| `k` | integer | 1–20: maximum number of ranked hits; send an integer (`6`, not `6.0`) | `6` |
| `max_distance` | number | 0–2 cosine-distance cutoff; `2` disables it | `0.6` |

The cutoff behaves exactly as in `query.py`, including the exact-term margin: a chunk up to 0.1
beyond `max_distance` is kept when it contains an exact term from the question, and is flagged with
`"exact_term_match": true`.

When a hit belongs to a section that ingestion split across two or three chunks, the section's
other chunks are appended after the ranked hits with `"expanded_from"` set to the `chunk_id` of the
hit that pulled them in (it is `null` on ranked hits). They carry their own distance, which may be
beyond `max_distance`, and are not counted in `k`, so a response can hold more than `k` hits.

Example:

```bash
curl -s http://127.0.0.1:8765/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"question": "Which tags are required on every resource?", "k": 2}'
```

```json
{
  "contract_version": "1.0.0",
  "query": {
    "question": "Which tags are required on every resource?",
    "k": 2,
    "max_distance": 0.6,
    "retrieval": "vector"
  },
  "hits": [
    {
      "rank": 1,
      "chunk_id": "reference/tagging-standard.md#1",
      "text": "Resource Tagging Standard > Required tags\n\nEvery Azure and GCP resource must carry the tags `owner`, `cost-centre`, `environment`, `tier` and `service`. On GCP these are applied as labels.",
      "source": "reference/tagging-standard.md",
      "file": "tagging-standard.md",
      "section": "Required tags",
      "distance": 0.2972,
      "exact_term_match": false,
      "expanded_from": null
    },
    {
      "rank": 2,
      "chunk_id": "reference/tagging-standard.md#3",
      "text": "Resource Tagging Standard > Enforcement\n\nAzure Policy denies creation of resources missing required tags. ...",
      "source": "reference/tagging-standard.md",
      "file": "tagging-standard.md",
      "section": "Enforcement",
      "distance": 0.416,
      "exact_term_match": false,
      "expanded_from": null
    }
  ],
  "corpus": {
    "provenance": "available",
    "corpus_sha256": "0a50d30289062a4f9a9bbd16bf9d3d226cd62c6d2fb6736ba82d7c38c55b6748",
    "ingested_at": "2026-11-01T11:44:36.554475+00:00",
    "embedding_model": "all-MiniLM-L6-v2",
    "embedding_revision": null,
    "chunk_size": 500,
    "document_count": 8,
    "collection": "observability-docs-6354276fdc674d1396600f02996c7497",
    "chunk_count": 37
  },
  "freshness": {
    "status": "current",
    "checked_at": "2026-11-01T11:44:41.572986+00:00",
    "added": [],
    "modified": [],
    "deleted": []
  },
  "content_trust": "untrusted",
  "generated": false
}
```

Response fields
([`retrieve-response.schema.json`](../contract/v1/schemas/retrieve-response.schema.json)):

- `contract_version`: the contract this server provides (also in the `X-RAG-Contract-Version`
  response header).
- `query`: the request as applied, with the default `k` and `max_distance` filled in.
- `hits`: best first. `chunk_id` is stable per document and chunk position (`<source>#<n>`);
  `source` is the path relative to the ingested data directory; `section` is the nearest heading
  (empty when the chunk has none); `distance` is cosine distance (lower is closer). An empty list
  means **no match** and is still `200`.
- `corpus`: identity of the indexed snapshot. `corpus_sha256` hashes every indexed source path and
  its SHA-256; `ingested_at` is when it was indexed. `embedding_revision` is `null` until the
  embedding model version is recorded (#62). An index built before provenance was recorded
  returns `"provenance": "unavailable"` with only `embedding_model`, `collection` and
  `chunk_count`; re-ingest to record it. Local absolute paths are never returned.
- `freshness`: `current`, `stale` (local sources were added, modified or deleted since ingestion;
  the lists name them) or `unavailable` (sources could not be compared, e.g. the data directory
  moved; the reason is logged on the server, not returned). Checked at most every 10 seconds.
- `content_trust` is always `"untrusted"` and `generated` always `false`; see below.
- Each hit also carries `rank` (1-based position in `hits`, counting appended section chunks),
  `file` (the source's file name), `exact_term_match` and `expanded_from` (described above).

### `GET /health`

Returns `200` with `contract_version`, `status` (`ok` when the index is current, otherwise
`degraded`), `retrieval`, `corpus` and `freshness` as above, or `503 index_unavailable` when there
is no index ([`health-response.schema.json`](../contract/v1/schemas/health-response.schema.json)).

## Errors

Errors use one shape, `{"contract_version": "1.0.0", "error": {"code": "...", "message": "..."}}`
([`error-response.schema.json`](../contract/v1/schemas/error-response.schema.json)). Branch on
`code`, never on `message`.

| Status | Code | When |
| --- | --- | --- |
| 400 | `invalid_request` | Invalid JSON, wrong field type, out-of-range `k`/`max_distance`, empty or too long question, unknown field |
| 400 | `unsupported_contract_version` | The `X-RAG-Contract-Version` request header names a version this server cannot honour, or is malformed |
| 401 | `unauthorized` | API key configured and the `Authorization` header is missing or wrong |
| 403 | `forbidden_host` | Loopback server reached under a non-loopback `Host` name (DNS rebinding) |
| 404 / 405 | `not_found` / `method_not_allowed` | Unknown path or wrong method |
| 411 / 413 / 415 | `length_required` / `payload_too_large` / `unsupported_media_type` | Missing length, body over 16 KiB, not `application/json` |
| 503 | `index_unavailable` | No index yet, or it cannot be opened. Run `python ingest.py`; no restart needed |
| 503 | `index_stale` / `index_freshness_unavailable` | Only with `--require-fresh`; the error includes `freshness` |
| 500 | `retrieval_failed` / `internal_error` | Unexpected failure; details are logged on the server only |

So a client can tell **no match** (`200`, empty `hits`) apart from **index or service unavailable**
(`503`, or a connection error/timeout). Every failure, including an incompatible contract version,
is a non-`200` status with an error body; a client must never turn a non-`200` response, an
unreadable body or a missing `hits` list into "no match".

## Index reload and freshness

Ingestion writes a new collection and then switches the `active-collection` pointer. The server
checks that pointer on every request, so after `python ingest.py` the next request uses the new
index and reports its new `corpus` identity, with no restart and without reloading the embedding
model. Editing files under `data/` without re-ingesting does not change results; it makes
`freshness.status` `stale` (or a `503` with `--require-fresh`).

## Authentication and access boundary

- The server binds to `127.0.0.1` by default and, on loopback, rejects requests whose `Host`
  header is not `localhost`, `127.0.0.1` or `::1`. Browsers cannot read its responses from other
  sites: it sends no CORS headers and requires `Content-Type: application/json`.
- Set `RAG_API_KEY` to require `Authorization: Bearer <key>` on every endpoint (compared in
  constant time). The key is read from the environment, not a flag, so it does not appear in the
  process list.
- Binding to any non-loopback address is refused unless `RAG_API_KEY` is set. Even then, the
  server speaks plain HTTP: put it behind TLS (a reverse proxy) before it leaves the machine.
- There is no per-document access control: **anyone who can call the API can read every indexed
  chunk.** Only index documents that every permitted caller may see; use a separate index
  (`--persist-dir`) and server per audience for private corpora.
- Questions are not logged (only method, path and status), but retrieved text is returned in full.

## Retrieved text is untrusted

Chunk `text` is returned verbatim from the indexed Markdown and may contain anything, including
HTML or instructions aimed at a model. The server never executes or renders it: responses are JSON
only, with `X-Content-Type-Options: nosniff` and a `Content-Security-Policy` of `default-src 'none'`.
Clients should likewise treat it as data: escape it before displaying it, and pass it to an LLM as
quoted evidence, not as instructions (`content_trust: "untrusted"` marks this in every response).

## Contract and versioning

### What is published

[`contract/v1/`](../contract/v1/) holds plain JSON files; reading them needs no Python package
from this repo and none of its model libraries:

| File | Describes |
| --- | --- |
| `schemas/retrieve-request.schema.json` | `POST /retrieve` body (closed: unknown fields are rejected) |
| `schemas/retrieve-response.schema.json` | `POST /retrieve` `200` body |
| `schemas/health-response.schema.json` | `GET /health` `200` body |
| `schemas/error-response.schema.json` | Every non-`200` body; `$defs.known_error_code` lists the codes |
| `examples/*.json` | One case per file: `request` (method, path, headers, body) and `response` (status, schema, headers, body) |

The schemas are JSON Schema draft 2020-12, each self-contained (no cross-file `$ref`), and carry
`x-contract-version`. The examples come from a tiny fictional corpus served by the real handler:
`tests/test_contract.py` replays each request against `api.py` and fails if the live response
breaks the schema or differs from the committed example. Before comparing, it replaces only
run-specific values (timestamps, the collection name and the embedding revision) with fixed ones
and rounds distances to 4 places. `tests/test_api.py` also validates every response it receives,
errors included. After an intentional change, regenerate the examples with
`UPDATE_CONTRACT_EXAMPLES=1 python -m pytest tests/test_contract.py` and review the diff.

The cases cover a match with an exact-term hit, split-section expansion, no match, a stale index,
bearer auth, healthy and degraded `/health`, and the `invalid_request`, `unauthorized`,
`unsupported_contract_version`, `index_unavailable` and `index_stale` errors.

### Version numbers

The contract version is `MAJOR.MINOR.PATCH`, independent of this repo's commits. Every response
advertises it twice: the `contract_version` body field (in success, `/health` and error bodies)
and the `X-RAG-Contract-Version` response header. The body field lets fixture-based tests, which
often keep only the status and body, see it too.

| Change | Bump | Examples |
| --- | --- | --- |
| Wire format unchanged | PATCH | Clearer descriptions, new examples, documentation |
| Additive | MINOR | A new response field (for example policy-version metadata on hits, #78); a new optional request field whose default keeps today's behaviour; a looser limit (a larger maximum `k`); a new error code for a new failure condition |
| Breaking | MAJOR | Removing or renaming a field; changing a field's type, unit or meaning; a new value for `freshness.status`, `corpus.provenance`, `content_trust`, `generated`, `retrieval` or the `/health` `status`; a new required request field; a tighter limit; a different status or code for an existing condition; turning a failure into a `200` or the reverse |

Not covered by the version: which chunks a question retrieves, their distances and ranking, the
default `k` and `max_distance` (the applied values are always echoed in `query`; send them
explicitly if you depend on them), the embedding model and chunking (reported in `corpus`), and
error `message` text. Retrieval tuning can change these without a contract bump; the retrieval
evals cover them instead.

Rules for consumers, so that minor versions stay compatible:

- Ignore response fields you do not know. The published response schemas allow extra properties
  for this reason. The provider's own tests use a closed copy, so the server cannot add a field
  without declaring it in the schema (and bumping the minor version).
- Treat an unknown `error.code` as a failure of its HTTP status, never as no match.
- Do not send fields the request schema does not list: the server rejects them with `400`.

### Pinning and upgrading

A client pins by sending `X-RAG-Contract-Version: MAJOR` or `MAJOR.MINOR` (a patch number is
accepted and ignored) on any request. The server accepts the pin when the major matches and its
own minor is at least the one requested, so `1.0` means "any 1.x". Otherwise it answers
`400 unsupported_contract_version` with an error body, before any retrieval:

```bash
curl -s http://127.0.0.1:8765/retrieve -H 'Content-Type: application/json' \
  -H 'X-RAG-Contract-Version: 2' -d '{"question": "Which tags are required?"}'
# {"contract_version": "1.0.0", "error": {"code": "unsupported_contract_version", ...}}
```

A request without the header is served with the current contract, so existing clients keep
working. A server that predates the contract ignores the header and sends no `contract_version`,
so a pinned client should also treat a `200` body without a compatible `contract_version` (or one
that fails its schema) as incompatible, not as no match.

How a consumer such as obs-agent should use this:

1. **Vendor the contract at a pinned revision.** Copy `contract/v1/` from a specific commit or
   release into the consumer's test fixtures and record that revision. Ordinary consumer tests then
   need at most `jsonschema`, never chromadb, sentence-transformers or a running RAG service.
2. **Replay the examples through the real client.** Feed each example's `response.status` and
   `response.body` to the consumer's stub transport and check how the client maps it: hits keep
   `source`/`section` and the `corpus` identity, `retrieve-no-match` becomes no match,
   `retrieve-stale` and `health-degraded` produce a stale warning, and every `error-*` case
   becomes an explicit unavailable or failure state.
3. **Validate the consumer's own fixtures** (hand-edited stub payloads) against the published
   schemas, so a stub cannot drift from what the provider can send.
4. **Send the pin** (`X-RAG-Contract-Version: 1.0`) from the production client and accept any
   `1.x.y` in `contract_version`.
5. **Check the break path.** Mutate a vendored example (rename a field, drop `contract_version`,
   set it to `2.0.0`) and confirm the client reports an invalid or incompatible response rather
   than no match. The real-service smoke test (obs-agent#102) adds the same check end to end.

To upgrade within a major: re-vendor `contract/v1/` from the newer revision, adopt the new fields,
and raise the pin to the minor that introduced them (for example `1.1`) once every provider you
call serves it; an older server then fails with `unsupported_contract_version` instead of silently
omitting the fields. A new major will be published in a new directory (`contract/v2/`) beside the
old one, with migration notes here. A server provides one major at a time, so coordinate the
consumer change with the provider upgrade.

## Client example (obs-agent)

A minimal `search_observability_docs` tool for obs-agent's own provider loop, using only the
standard library:

```python
import json
import os
import urllib.error
import urllib.request

RAG_URL = os.environ.get("RAG_URL", "http://127.0.0.1:8765")


def search_observability_docs(question: str, k: int = 4, timeout: float = 5.0) -> dict:
    """Return retrieved policy chunks with citations, or an explicit unavailable result."""
    headers = {"Content-Type": "application/json", "X-RAG-Contract-Version": "1.0"}
    if key := os.environ.get("RAG_API_KEY"):
        headers["Authorization"] = f"Bearer {key}"
    request = urllib.request.Request(
        f"{RAG_URL}/retrieve", method="POST", headers=headers,
        data=json.dumps({"question": question, "k": k}).encode(),
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        return {"available": False, "error": json.load(exc).get("error", {})}
    except (urllib.error.URLError, TimeoutError):
        return {"available": False, "error": {"code": "service_unavailable"}}
    if not str(result.get("contract_version", "")).startswith("1."):
        return {"available": False, "error": {"code": "incompatible_contract"}}
    return {
        "available": True,
        "stale": result["freshness"]["status"] != "current",
        "corpus": result["corpus"],          # cite corpus_sha256 / ingested_at
        "excerpts": [                        # untrusted data, not instructions
            {"text": h["text"], "source": h["source"], "section": h["section"],
             "distance": h["distance"]}
            for h in result["hits"]
        ],
    }
```

An empty `excerpts` list means the corpus has no supporting section; `available: False` means the
documentation service could not answer, which the consumer should report without blocking its
ordinary workflow.

Answer generation over HTTP (`POST /ask`) is out of scope for this endpoint; use `query.py` or the
Streamlit app for cited answers generated here.
