# Retrieval MCP server

`mcp_server.py` exposes retrieval as a read-only [Model Context Protocol](https://modelcontextprotocol.io)
server, so Claude Code, Claude Desktop and other MCP clients (including obs-agent) can search the
indexed documents as a tool.

It is the [HTTP API](http-api.md) over a different transport, not a second API:

- one tool, `retrieve`, whose arguments are the `POST /retrieve` request body;
- the same engine (`RetrievalService` in `api.py`, so the same `RAGEngine.retrieve()`, cutoff,
  exact-term margin, section completion and policy metadata);
- the same response, returned as the tool's structured content. Both schemas are the published
  contract files in [`contract/v1/`](../contract/v1/), served verbatim as the tool's
  `inputSchema` and `outputSchema`;
- the same errors, returned as MCP tool errors that carry the HTTP error body.

It is **retrieval-only**: it never calls an LLM provider and needs no provider key. The client's
own model writes and cites the answer.

## Install and run

```bash
python -m pip install -r requirements-mcp.txt   # core install plus the mcp SDK (2.x)
python ingest.py                                # build or update the index
python mcp_server.py                            # stdio; normally your MCP client starts it
```

The `mcp` package is optional; `requirements.txt` does not include it. The server speaks stdio
only and logs to stderr, because stdout carries the protocol.

| Flag | Purpose | Default |
| --- | --- | --- |
| `--persist-dir` | Chroma index directory | `chroma_db` next to `mcp_server.py` (not the working directory, which MCP clients do not set reliably) |
| `--data-dir` | Source directory to compare for freshness | the directory that was ingested |
| `--retrieval` | `vector` or `hybrid` (BM25 + vector), same as `api.py` | `vector` |
| `--require-fresh` | Return an `index_stale` tool error instead of results when sources changed since ingestion | off |

The embedding model loads in the background at startup, so the client's handshake is not delayed;
a call made before it finishes waits for it. As with the HTTP API, re-running `python ingest.py`
switches the server to the new index on the next call without a restart.

## Claude Code

Use absolute paths, and the Python from the environment where you installed `requirements-mcp.txt`:

```bash
claude mcp add rag-observability -- \
  /path/to/rag-observability/.venv/bin/python /path/to/rag-observability/mcp_server.py
```

Add `--scope project` to write the entry to `.mcp.json` and share it with a project, and put
server flags after the script path (for example `... mcp_server.py --retrieval hybrid`). Check it
with `claude mcp list`, or `/mcp` inside a session.

## Claude Desktop

Add the server to `claude_desktop_config.json` (macOS:
`~/Library/Application Support/Claude/claude_desktop_config.json`; Windows:
`%APPDATA%\Claude\claude_desktop_config.json`) and restart Claude Desktop:

```json
{
  "mcpServers": {
    "rag-observability": {
      "command": "/path/to/rag-observability/.venv/bin/python",
      "args": ["/path/to/rag-observability/mcp_server.py"]
    }
  }
}
```

## The `retrieve` tool

Arguments ([`retrieve-request.schema.json`](../contract/v1/schemas/retrieve-request.schema.json)):

| Argument | Type | Rules | Default |
| --- | --- | --- | --- |
| `question` | string | 1-2000 characters after trimming, no NUL | required |
| `k` | integer | 1-20, ranked hits (section completion can append more) | 6 |
| `max_distance` | number | 0-2 cosine-distance cutoff; 2 disables it | 0.6 |

Unknown arguments are rejected, exactly as unknown fields are over HTTP. The retrieval mode is a
server flag (`--retrieval`), as it is for `api.py`; `query.retrieval` in the result reports it.

The result's `structuredContent` is the `POST /retrieve` response body and validates against
[`retrieve-response.schema.json`](../contract/v1/schemas/retrieve-response.schema.json): `hits`
(text, source, section, distance, exact-term and section-expansion flags, policy status),
`corpus` identity, source `freshness`, `content_trust: "untrusted"` and `generated: false`. The
same JSON is repeated as text content for clients that ignore structured content. See
[`POST /retrieve`](http-api.md#post-retrieve) for what each field means and
[policy applicability](http-api.md#policy-applicability-contract-11) for `policy.status`.

An empty `hits` list is a successful **no match**, never an error.

### Errors

Every failure is a tool result with `isError: true` whose text content is the HTTP error body
([`error-response.schema.json`](../contract/v1/schemas/error-response.schema.json)), for example:

```json
{"contract_version": "1.1.0", "error": {"code": "index_unavailable", "message": "No index found. Run python ingest.py, then retry."}}
```

The codes are the HTTP ones that can apply: `invalid_request`, `index_unavailable`,
`index_stale` and `index_freshness_unavailable` (with `--require-fresh`, including `freshness`),
`retrieval_failed` and `internal_error` (details logged on the server only), and `not_found` for
an unknown tool name. Branch on `code`, never on `message`. HTTP-only codes (authentication, host
checks, content type, body size) do not arise over stdio.

## Retrieved text is untrusted

Each hit's `text` is verbatim document content. A document can contain instructions aimed at an AI
assistant: the [adversarial scenarios](guide.md#adversarial-scenarios) include a paging guide that
tells the assistant to ignore the user's question and a notice that tells it to report a false
availability target. The MCP server returns such text unchanged, as it should, so:

- the tool description, the server instructions and every result (`content_trust: "untrusted"`)
  tell the model to treat hit text as quoted evidence, never as instructions;
- an agent built on this tool should keep the same rule: cite the text, check it against the
  question, and never let it trigger other tool calls by itself;
- only index documents you trust the client's model to read.

## How this relates to the HTTP API

| | HTTP API (`api.py`) | MCP server (`mcp_server.py`) |
| --- | --- | --- |
| Transport | HTTP on loopback (or beyond, with `RAG_API_KEY`) | stdio, started by the MCP client |
| Clients | Any HTTP client, including obs-agent's | MCP clients: Claude Code, Claude Desktop, agents |
| Request | `POST /retrieve` JSON body | `retrieve` tool arguments (same schema) |
| Response | `200` JSON body | `structuredContent` (same schema) |
| Errors | non-`200` status + error body | `isError: true` + the same error body |
| Health | `GET /health` | not exposed; `corpus` and `freshness` come with every result |
| Contract pinning | `X-RAG-Contract-Version` header | not available; check `contract_version` in each result |
| Auth | Optional bearer token | None needed: the client launches the process locally |
| Dependencies | Standard library on top of `requirements.txt` | `requirements-mcp.txt` (the `mcp` SDK) |

Both share one implementation, so a change to retrieval or to the contract reaches both. Tests in
[`tests/test_mcp_server.py`](../tests/test_mcp_server.py) drive the server through the SDK's
in-process client and validate every result against the same closed contract schemas as the HTTP
tests.
