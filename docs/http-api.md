# Retrieval HTTP API

`api.py` serves the existing retrieval engine (`RAGEngine.retrieve()`) over local HTTP so other
tools (for example obs-agent) can fetch supporting chunks without the CLI or UI.

It is **retrieval-only**: it never calls Claude, DeepSeek or any other LLM, needs no provider key,
and returns no generated answer. The caller generates and cites its own answer from the returned
chunks. It uses only the Python standard library on top of `requirements.txt`.

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

Request (`Content-Type: application/json`, body at most 16 KiB; unknown fields are rejected):

| Field | Type | Rules | Default |
| --- | --- | --- | --- |
| `question` | string | 1–2000 characters after trimming; no NUL | required |
| `k` | integer | 1–20: maximum number of hits | `6` |
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

Response fields:

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

### `GET /health`

Returns `200` with `status` (`ok` when the index is current, otherwise `degraded`), `retrieval`,
`corpus` and `freshness` as above, or `503 index_unavailable` when there is no index.

## Errors

Errors use one shape: `{"error": {"code": "...", "message": "..."}}`.

| Status | Code | When |
| --- | --- | --- |
| 400 | `invalid_request` | Invalid JSON, wrong field type, out-of-range `k`/`max_distance`, empty or too long question, unknown field |
| 401 | `unauthorized` | API key configured and the `Authorization` header is missing or wrong |
| 403 | `forbidden_host` | Loopback server reached under a non-loopback `Host` name (DNS rebinding) |
| 404 / 405 | `not_found` / `method_not_allowed` | Unknown path or wrong method |
| 411 / 413 / 415 | `length_required` / `payload_too_large` / `unsupported_media_type` | Missing length, body over 16 KiB, not `application/json` |
| 503 | `index_unavailable` | No index yet, or it cannot be opened. Run `python ingest.py`; no restart needed |
| 503 | `index_stale` / `index_freshness_unavailable` | Only with `--require-fresh`; the error includes `freshness` |
| 500 | `retrieval_failed` / `internal_error` | Unexpected failure; details are logged on the server only |

So a client can tell **no match** (`200`, empty `hits`) apart from **index or service unavailable**
(`503`, or a connection error/timeout).

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
    headers = {"Content-Type": "application/json"}
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
