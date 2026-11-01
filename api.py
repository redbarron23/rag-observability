#!/usr/bin/env python3
"""Retrieval-only HTTP API over the local index.

Usage
-----
    python api.py                          # http://127.0.0.1:8765
    python api.py --retrieval hybrid --port 9000
    RAG_API_KEY=... python api.py          # require "Authorization: Bearer ..."

Endpoints
---------
    POST /retrieve   {"question": "...", "k": 6, "max_distance": 0.6}
    GET  /health     index identity and source freshness

Every response carries the contract version (`contract_version` in the body and
the X-RAG-Contract-Version header). A client may pin one by sending
X-RAG-Contract-Version: MAJOR[.MINOR]; an incompatible pin is rejected with
400 unsupported_contract_version. The JSON Schemas are in contract/v1/.

POST /retrieve returns the chunks RAGEngine.retrieve() selects, with their
source, section and cosine distance, plus the indexed corpus identity and
freshness. It never calls an LLM provider, so it needs no provider key; the
caller generates any answer itself. Uses only the standard library on top of
the existing requirements. See docs/http-api.md for the contract and policy.
"""

import argparse
import hmac
import ipaddress
import json
import logging
import math
import os
import re
import socket
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from chromadb.errors import ChromaError

from ingest import EMBED_MODEL, check_freshness
from query import (
    COLLECTION_NAME,
    DEFAULT_RETRIEVAL,
    MAX_DISTANCE,
    N_RESULTS,
    PERSIST_DIR,
    RETRIEVAL_MODES,
    RAGEngine,
    kept_by_exact_term,
)

logger = logging.getLogger("rag-api")

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
MAX_QUESTION_CHARS = 2000
MAX_K = 20
MAX_BODY_BYTES = 16 * 1024
FRESHNESS_TTL_SECONDS = 10.0
API_KEY_ENV = "RAG_API_KEY"
REQUEST_FIELDS = {"question", "k", "max_distance"}
LOOPBACK_HOST_NAMES = {"localhost", "127.0.0.1", "::1"}

# The wire contract published in contract/v1/ (docs/http-api.md#contract-versioning).
# MINOR: additive changes only; MAJOR: anything a pinned client could break on.
CONTRACT_VERSION = "1.1.0"
CONTRACT_HEADER = "X-RAG-Contract-Version"
_PINNED_VERSION_RE = re.compile(r"(\d{1,4})(?:\.(\d{1,4})(?:\.\d{1,4})?)?")


class ApiError(Exception):
    """An error with an HTTP status, a stable code and a client-safe message."""

    def __init__(self, status: int, code: str, message: str, details: dict | None = None):
        super().__init__(message)
        self.status, self.code, self.message, self.details = status, code, message, details


def validate_request(body: object) -> tuple[str, int, float]:
    """Return (question, k, max_distance) or raise a 400 ApiError."""
    def invalid(message: str) -> ApiError:
        return ApiError(400, "invalid_request", message)

    if not isinstance(body, dict):
        raise invalid("The request body must be a JSON object.")
    unknown = sorted(set(body) - REQUEST_FIELDS)
    if unknown:
        raise invalid(f"Unknown field(s): {', '.join(unknown)}.")
    question = body.get("question")
    if not isinstance(question, str):
        raise invalid("'question' must be a string.")
    question = question.strip()
    if not question:
        raise invalid("'question' must not be empty.")
    if len(question) > MAX_QUESTION_CHARS:
        raise invalid(f"'question' must be at most {MAX_QUESTION_CHARS} characters.")
    if "\x00" in question:
        raise invalid("'question' must not contain NUL characters.")
    k = body.get("k", N_RESULTS)
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= MAX_K:
        raise invalid(f"'k' must be an integer from 1 to {MAX_K}.")
    max_distance = body.get("max_distance", MAX_DISTANCE)
    if (isinstance(max_distance, bool) or not isinstance(max_distance, int | float)
            or not math.isfinite(max_distance) or not 0 <= max_distance <= 2):
        raise invalid("'max_distance' must be a number from 0 to 2 (2 disables the cutoff).")
    return question, k, float(max_distance)


def check_contract_version(requested: str | None) -> None:
    """Reject a pinned contract version this server cannot honour.

    A client pins MAJOR or MAJOR.MINOR (a patch number is accepted and ignored).
    The pin is compatible when the major matches and the server's minor is at
    least the requested one, so a client that needs fields added in 1.1 fails
    explicitly against a 1.0 server instead of silently missing them.
    """
    if requested is None:
        return
    major, minor, _ = (int(part) for part in CONTRACT_VERSION.split("."))
    match = _PINNED_VERSION_RE.fullmatch(requested.strip())
    if not match:
        raise ApiError(400, "unsupported_contract_version",
                       f"{CONTRACT_HEADER} must be MAJOR or MAJOR.MINOR, "
                       f"for example {major} or {major}.{minor}.")
    if int(match[1]) != major or int(match[2] or 0) > minor:
        raise ApiError(400, "unsupported_contract_version",
                       f"This server provides contract version {CONTRACT_VERSION}, "
                       f"which is not compatible with requested version {match[0]}.")


def corpus_identity(engine: RAGEngine) -> dict:
    """The indexed snapshot's identity, without the manifest's local paths."""
    identity = {"collection": engine.collection_name, "chunk_count": engine.collection.count()}
    try:
        manifest = json.loads((engine.collection.metadata or {}).get("corpus_manifest", "null"))
        if manifest and manifest.get("schema_version") == 1:
            return {
                "provenance": "available",
                "corpus_sha256": manifest["corpus_sha256"],
                "ingested_at": manifest["ingested_at"],
                "embedding_model": manifest["embedding_model"],
                "embedding_revision": manifest.get("embedding_revision"),
                "chunk_size": manifest.get("chunk_size"),
                "document_count": len(manifest["sources"]),
                **identity,
            }
    except (ValueError, KeyError, TypeError, AttributeError):
        pass  # Legacy or malformed metadata must not masquerade as known provenance.
    return {"provenance": "unavailable", "embedding_model": EMBED_MODEL, **identity}


class RetrievalService:
    """Holds one embedder and the engine for the active index.

    The embedder loads once. The active-collection pointer is checked on every
    request, so after `python ingest.py` the next request uses the new index
    without a restart (the embedder is reused).
    """

    def __init__(self, persist_dir: str = PERSIST_DIR, retrieval: str = DEFAULT_RETRIEVAL,
                 data_dir: str | None = None, require_fresh: bool = False,
                 engine_factory=None, embedder_factory=None):
        if retrieval not in RETRIEVAL_MODES:
            raise ValueError(f"Unknown retrieval mode {retrieval!r}")
        self.persist_dir, self.retrieval = persist_dir, retrieval
        self.data_dir, self.require_fresh = data_dir, require_fresh
        self._engine_factory = engine_factory or (
            lambda embedder: RAGEngine(persist_dir, retrieval=retrieval, embedder=embedder))
        self._embedder_factory = embedder_factory or _load_embedder
        self._embedder = None
        self._engine: RAGEngine | None = None
        self._freshness: tuple[float, str, dict] | None = None
        self._lock = threading.Lock()

    def load_embedder(self) -> None:
        """Load the embedding model now rather than on the first request."""
        with self._lock:
            if self._embedder is None:
                self._embedder = self._embedder_factory()

    def _active_name(self) -> str | None:
        root = Path(self.persist_dir)
        if not root.is_dir():
            return None
        try:
            return (root / "active-collection").read_text(encoding="utf-8")
        except FileNotFoundError:
            return COLLECTION_NAME
        except (OSError, UnicodeError):
            return None

    def engine(self) -> RAGEngine:
        """The engine for the active index, reloaded when the pointer changes.

        Call with the lock held.
        """
        name = self._active_name()
        if name is None:
            self._engine = None
            raise ApiError(503, "index_unavailable",
                           "No index found. Run python ingest.py, then retry.")
        if self._engine is None or self._engine.collection_name != name:
            if self._embedder is None:
                self._embedder = self._embedder_factory()
            try:
                self._engine = self._engine_factory(self._embedder)
            except (ChromaError, OSError, ValueError) as exc:
                self._engine = None
                logger.warning("Could not open the index at %s: %s", self.persist_dir, exc)
                raise ApiError(503, "index_unavailable",
                               "The index could not be opened. Run python ingest.py, "
                               "then retry.") from None
            self._freshness = None
        return self._engine

    def freshness(self, engine: RAGEngine) -> dict:
        """Whether local sources changed since ingestion, cached briefly."""
        now = time.monotonic()
        cached = self._freshness
        if cached and cached[1] == engine.collection_name and now - cached[0] < FRESHNESS_TTL_SECONDS:
            return cached[2]
        result = check_freshness(self.persist_dir, self.data_dir)
        checked_at = datetime.now(timezone.utc).isoformat()
        if result["status"] == "unavailable":
            # The reason can name local paths; log it instead of returning it.
            logger.warning("Freshness unavailable: %s", result.get("reason"))
            fresh = {"status": "unavailable", "checked_at": checked_at,
                     "reason": "Source documents could not be compared with the index."}
        else:
            fresh = {"status": result["status"], "checked_at": checked_at,
                     "added": result["added"], "modified": result["modified"],
                     "deleted": result["deleted"]}
        self._freshness = (now, engine.collection_name, fresh)
        return fresh

    def _check_fresh(self, fresh: dict) -> None:
        if not self.require_fresh or fresh["status"] == "current":
            return
        if fresh["status"] == "stale":
            raise ApiError(503, "index_stale",
                           "Source documents changed since ingestion. Run python ingest.py.",
                           {"freshness": fresh})
        raise ApiError(503, "index_freshness_unavailable",
                       "Index freshness could not be verified.", {"freshness": fresh})

    def health(self) -> dict:
        with self._lock:
            engine = self.engine()
            corpus, fresh = corpus_identity(engine), self.freshness(engine)
        return {"contract_version": CONTRACT_VERSION,
                "status": "ok" if fresh["status"] == "current" else "degraded",
                "retrieval": self.retrieval, "corpus": corpus, "freshness": fresh}

    def retrieve(self, body: object) -> dict:
        question, k, max_distance = validate_request(body)
        with self._lock:
            engine = self.engine()
            fresh = self.freshness(engine)
            self._check_fresh(fresh)
            try:
                chunks = engine.retrieve(question, n_results=k, max_distance=max_distance)
                corpus = corpus_identity(engine)
            except Exception as exc:
                logger.exception("Retrieval failed")
                self._engine = None  # e.g. the collection vanished; reopen next time
                raise ApiError(500, "retrieval_failed",
                               "Retrieval failed; see the server log.") from exc
        hits = [{
            "rank": rank,
            "chunk_id": c["id"],
            "text": c["text"],
            "source": c["source"],
            "file": c["file"],
            "section": c["section"],
            "distance": c["distance"],
            # Kept only by the exact-term margin beyond max_distance.
            "exact_term_match": kept_by_exact_term(c, max_distance),
            # Appended because a ranked hit is in the same split section (#72).
            "expanded_from": c.get("expanded_from"),
            # Policy applicability on the server's current UTC date (#78);
            # independent of source-file freshness.
            "policy": c.get("policy"),
        } for rank, c in enumerate(chunks, start=1)]
        return {
            "contract_version": CONTRACT_VERSION,
            "query": {"question": question, "k": k, "max_distance": max_distance,
                      "retrieval": self.retrieval},
            "hits": hits,
            "corpus": corpus,
            "freshness": fresh,
            "content_trust": "untrusted",
            "generated": False,
        }


def _load_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL)


def is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def make_handler(service: RetrievalService, api_key: str | None, check_host: bool):
    """Build a request handler bound to the service and auth settings."""
    expected = f"Bearer {api_key}".encode() if api_key else None

    class Handler(BaseHTTPRequestHandler):
        server_version = "rag-observability"
        sys_version = ""
        protocol_version = "HTTP/1.1"

        def log_message(self, format, *args):
            logger.info("%s %s", self.address_string(), format % args)

        def _send(self, status: int, payload: dict, extra: dict | None = None) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            # Document text is untrusted data: JSON only, never sniffed or rendered.
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'none'; sandbox")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.send_header(CONTRACT_HEADER, CONTRACT_VERSION)
            for name, value in (extra or {}).items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)

        def _error(self, err: ApiError) -> None:
            payload = {"contract_version": CONTRACT_VERSION,
                       "error": {"code": err.code, "message": err.message, **(err.details or {})}}
            extra = {"WWW-Authenticate": "Bearer"} if err.status == 401 else None
            if err.status in (400, 401, 404, 405, 411, 413, 415):
                self.close_connection = True
            self._send(err.status, payload, extra)

        def _authorize(self) -> None:
            if check_host:
                # Reject DNS-rebinding requests that reach loopback under another name.
                host = (self.headers.get("Host") or "").strip()
                name = host[1:].split("]")[0] if host.startswith("[") else host.rsplit(":", 1)[0]
                if name.lower() not in LOOPBACK_HOST_NAMES:
                    raise ApiError(403, "forbidden_host", "Host header must name a loopback address.")
            if expected is not None:
                supplied = (self.headers.get("Authorization") or "").encode()
                if not hmac.compare_digest(supplied, expected):
                    raise ApiError(401, "unauthorized", "Missing or invalid API key.")

        def _read_json(self) -> object:
            content_type = (self.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            if content_type != "application/json":
                raise ApiError(415, "unsupported_media_type", "Send Content-Type: application/json.")
            if self.headers.get("Transfer-Encoding"):
                raise ApiError(411, "length_required", "Send a Content-Length; chunked bodies are not accepted.")
            try:
                length = int(self.headers.get("Content-Length", ""))
            except ValueError:
                raise ApiError(411, "length_required", "Send a Content-Length header.") from None
            if length < 0 or length > MAX_BODY_BYTES:
                raise ApiError(413, "payload_too_large", f"Request body must be at most {MAX_BODY_BYTES} bytes.")
            try:
                return json.loads(self.rfile.read(length).decode("utf-8"))
            except (UnicodeDecodeError, ValueError):
                raise ApiError(400, "invalid_request", "The request body must be valid JSON.") from None

        def _route(self, method: str) -> None:
            path = self.path.split("?", 1)[0]
            try:
                self._authorize()
                check_contract_version(self.headers.get(CONTRACT_HEADER))
                if path == "/retrieve":
                    if method != "POST":
                        raise ApiError(405, "method_not_allowed", "Use POST /retrieve.")
                    self._send(200, service.retrieve(self._read_json()))
                elif path == "/health":
                    if method != "GET":
                        raise ApiError(405, "method_not_allowed", "Use GET /health.")
                    self._send(200, service.health())
                else:
                    raise ApiError(404, "not_found", "Unknown endpoint. Use POST /retrieve or GET /health.")
            except ApiError as err:
                self._error(err)
            except Exception:
                logger.exception("Unhandled API error")
                self._error(ApiError(500, "internal_error", "Internal error; see the server log."))

        def do_GET(self):
            self._route("GET")

        def do_POST(self):
            self._route("POST")

        def do_PUT(self):
            self._route("PUT")

        def do_DELETE(self):
            self._route("DELETE")

    return Handler


class _IPv6Server(ThreadingHTTPServer):
    address_family = socket.AF_INET6


def make_server(service: RetrievalService, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT,
                api_key: str | None = None) -> ThreadingHTTPServer:
    """Create (but do not start) the HTTP server.

    Binding beyond loopback requires an API key.
    """
    loopback = is_loopback(host)
    if not loopback and not api_key:
        raise ValueError(f"Refusing to bind {host} without an API key; set {API_KEY_ENV}.")
    server_cls = _IPv6Server if ":" in host else ThreadingHTTPServer
    server = server_cls((host, port), make_handler(service, api_key, check_host=loopback))
    server.daemon_threads = True
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval-only HTTP API (no answer generation)")
    parser.add_argument("--host", default=DEFAULT_HOST,
                        help=f"Bind address (default: {DEFAULT_HOST}; others need {API_KEY_ENV})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--persist-dir", default=PERSIST_DIR)
    parser.add_argument("--data-dir", default=None,
                        help="Source directory for freshness checks (default: the one ingested)")
    parser.add_argument("--retrieval", default=DEFAULT_RETRIEVAL, choices=RETRIEVAL_MODES)
    parser.add_argument("--require-fresh", action="store_true",
                        help="Reject retrieval (503) when sources changed since ingestion")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)  # no progress bars

    api_key = os.environ.get(API_KEY_ENV) or None
    service = RetrievalService(args.persist_dir, args.retrieval, args.data_dir, args.require_fresh)
    try:
        server = make_server(service, args.host, args.port, api_key)
    except (ValueError, OSError) as exc:
        print(exc, file=sys.stderr)
        sys.exit(2)
    logger.info("Loading embedding model %s...", EMBED_MODEL)
    service.load_embedder()
    try:
        service.health()
    except ApiError as err:
        logger.warning("%s The server is running and will open the index when it exists.",
                       err.message)
    auth = "API key required" if api_key else "no authentication (loopback only)"
    logger.info("Serving retrieval on http://%s:%d (%s)", args.host, server.server_port, auth)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
