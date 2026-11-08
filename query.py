#!/usr/bin/env python3
"""Query the observability RAG system.

Usage
-----
    python query.py "What are the coverage targets for production Azure resources?"
    python query.py --n-results 5 "What is the observability framework scoring model?"
    python query.py --verbose "How does GCP metric collection differ from Azure?"

This retrieves the top-k relevant chunks from Chroma, then asks an LLM
to answer the question using ONLY those chunks as sources (cited).

Default provider is Anthropic (Claude). Set LLM_PROVIDER=deepseek to use DeepSeek.
On the Anthropic path each chunk is sent as a document with Claude's native
citations enabled; DeepSeek is prompted to write its citations in the answer.
"""

import argparse
import json
import logging
import math
import os
import re
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, date, datetime
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import chromadb
from sentence_transformers import SentenceTransformer

import policy
from ingest import EMBED_MODEL

# Keep this in query.py: consumer images copy this module explicitly.
_current_trace_span = ContextVar("rag_trace_span", default=None)
_trace_logger = logging.getLogger("rag.tracing")


@lru_cache(maxsize=1)
def _trace_provider():
    """Configure a private provider once; leave the host application's provider alone."""
    exporter = os.environ.get("OTEL_TRACES_EXPORTER", "").strip().lower()
    if exporter in ("", "none") or os.environ.get("OTEL_SDK_DISABLED", "").lower() == "true":
        return None
    try:
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

        if exporter == "console":
            import sys
            target = ConsoleSpanExporter(out=sys.stderr)
        elif exporter == "otlp":
            protocol = os.environ.get("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL",
                                      os.environ.get("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf"))
            if protocol != "http/protobuf":
                raise ValueError("Only OTLP HTTP/protobuf is supported")
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            target = OTLPSpanExporter()
        else:
            raise ValueError("Choose console, otlp or none")
        provider = TracerProvider(resource=Resource.create({
            "service.name": os.environ.get("OTEL_SERVICE_NAME", "rag-observability"),
        }))
        provider.add_span_processor(BatchSpanProcessor(target))
        return provider
    except Exception as exc:  # noqa: BLE001 - optional telemetry must not break a question
        _trace_logger.warning("Tracing disabled (%s); question processing continues.", type(exc).__name__)
        return None



_extracting_context = ContextVar("rag_extracting_context", default=False)


class _ContextLogFilter(logging.Filter):
    def filter(self, record):
        if _extracting_context.get():
            # The SDK's tracestate parser can log invalid raw header values.
            record.msg, record.args = "Invalid trace context metadata ignored.", ()
        return True


@lru_cache(maxsize=1)
def _propagator():
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
    logging.getLogger("opentelemetry.trace.span").addFilter(_ContextLogFilter())
    return TraceContextTextMapPropagator()


@contextmanager
def incoming_trace(headers):
    """Isolate each authenticated HTTP request; accept only standard W3C trace context."""
    token = None
    extracting = _extracting_context.set(True)
    try:
        if _trace_provider() is not None:
            from opentelemetry import context
            carrier = {key: headers[key] for key in ("traceparent", "tracestate") if headers.get(key)}
            token = context.attach(_propagator().extract(carrier, context=context.Context()))
    except Exception:  # noqa: BLE001 - correlation must not break retrieval
        _trace_logger.warning("Could not extract trace context.")
    finally:
        _extracting_context.reset(extracting)
    try:
        yield
    finally:
        if token is not None:
            try:
                context.detach(token)
            except Exception:  # noqa: BLE001 - preserve the request result on SDK failure
                _trace_logger.warning("Could not detach trace context.")


def _tracer():
    provider = _trace_provider()
    return provider.get_tracer("rag-observability") if provider is not None else None


def trace_attributes(values: dict) -> None:
    span = _current_trace_span.get()
    if span is not None:
        try:
            span.set_attributes({key: value for key, value in values.items() if value is not None})
        except Exception:  # noqa: BLE001 - a telemetry failure cannot change application behavior
            _trace_logger.warning("Could not record trace attributes.")


@contextmanager
def trace_span(name: str):
    """A span/decorator that never captures question, document, answer or exception text."""
    try:
        tracer = _tracer()
        manager = (tracer.start_as_current_span(name, record_exception=False,
                   set_status_on_exception=False) if tracer is not None else None)
        span = manager.__enter__() if manager is not None else None
    except Exception:  # noqa: BLE001 - setup failures must not prevent the wrapped operation
        _trace_logger.warning("Could not start trace span.")
        manager, span = None, None
    token = _current_trace_span.set(span)
    started = perf_counter()
    try:
        yield span
    except BaseException as exc:
        trace_attributes({"error.type": type(exc).__name__})
        if span is not None:
            try:
                from opentelemetry.trace import StatusCode
                span.set_status(StatusCode.ERROR)
            except Exception:  # noqa: BLE001 - preserve the original application exception
                _trace_logger.warning("Could not record trace error status.")
        raise
    finally:
        trace_attributes({"rag.duration_ms": (perf_counter() - started) * 1000})
        _current_trace_span.reset(token)
        if manager is not None:
            try:
                manager.__exit__(None, None, None)
            except Exception:  # noqa: BLE001 - exporter/span failures must not mask the result
                _trace_logger.warning("Could not finish trace span.")


def trace_generation(meta: dict) -> None:
    usage = meta.get("usage") or {}
    trace_attributes({
        "gen_ai.provider.name": meta.get("provider"),
        "gen_ai.request.model": meta.get("requested_model"),
        "gen_ai.response.model": meta.get("response_model"),
        "gen_ai.usage.input_tokens": usage.get("input_tokens"),
        "gen_ai.usage.output_tokens": usage.get("output_tokens"),
        "rag.generation.stop_reason": meta.get("stop_reason"),
        "rag.generation.refused": (meta["stop_reason"] == "refusal"
                                   if meta.get("stop_reason") is not None else None),
        "rag.generation.fallback_used": meta.get("fallback_used"),
    })


# ── Metrics ────────────────────────────────────────────────────────────
# Opt-in like tracing, through the standard OTEL_METRICS_EXPORTER switch, and
# exported with the same OTLP settings. Attributes are a fixed, small set:
# retrieval mode, outcome, provider, model IDs, token type and error class.
# Never questions, prompts, document text, answers, source paths or chunk IDs.

# Bucket boundaries advised by the OTel GenAI semantic conventions.
_DURATION_BUCKETS = (0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12,
                     10.24, 20.48, 40.96, 81.92)
_TOKEN_BUCKETS = (1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576,
                  4194304, 16777216, 67108864)


@lru_cache(maxsize=1)
def _meter_provider():
    """A private MeterProvider, or None when metrics are off (the default)."""
    exporter = os.environ.get("OTEL_METRICS_EXPORTER", "").strip().lower()
    if exporter in ("", "none") or os.environ.get("OTEL_SDK_DISABLED", "").lower() == "true":
        return None
    try:
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import (
            ConsoleMetricExporter,
            PeriodicExportingMetricReader,
        )
        from opentelemetry.sdk.resources import Resource

        if exporter == "console":
            import sys
            target = ConsoleMetricExporter(out=sys.stderr)
        elif exporter == "otlp":
            protocol = os.environ.get("OTEL_EXPORTER_OTLP_METRICS_PROTOCOL",
                                      os.environ.get("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf"))
            if protocol != "http/protobuf":
                raise ValueError("Only OTLP HTTP/protobuf is supported")
            from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
            target = OTLPMetricExporter()
        else:
            raise ValueError("Choose console, otlp or none")
        return MeterProvider(
            metric_readers=[PeriodicExportingMetricReader(target)],
            resource=Resource.create({
                "service.name": os.environ.get("OTEL_SERVICE_NAME", "rag-observability"),
            }),
        )
    except Exception as exc:  # noqa: BLE001 - optional telemetry must not break a question
        _trace_logger.warning("Metrics disabled (%s); question processing continues.", type(exc).__name__)
        return None


@lru_cache(maxsize=1)
def _instruments():
    """The metric instruments, or None when metrics are off."""
    provider = _meter_provider()
    if provider is None:
        return None
    try:
        meter = provider.get_meter("rag-observability")
        return {
            "operation_duration": meter.create_histogram(
                "gen_ai.client.operation.duration", unit="s",
                description="Duration of GenAI provider calls",
                explicit_bucket_boundaries_advisory=_DURATION_BUCKETS),
            "token_usage": meter.create_histogram(
                "gen_ai.client.token.usage", unit="{token}",
                description="Input and output tokens per provider call, as reported by the provider",
                explicit_bucket_boundaries_advisory=_TOKEN_BUCKETS),
            "retrieval_duration": meter.create_histogram(
                "rag.retrieval.duration", unit="s",
                description="Duration of retrieval: embedding, search, section completion, policy status",
                explicit_bucket_boundaries_advisory=_DURATION_BUCKETS),
            "chunks": meter.create_counter(
                "rag.retrieval.chunks", unit="{chunk}",
                description="Retrieved chunks by what the distance cutoff did with them"),
            "questions": meter.create_counter(
                "rag.questions", unit="{question}",
                description="Answered-question operations by outcome"),
            "refusals": meter.create_counter(
                "rag.generation.refusals", unit="{refusal}",
                description="Provider calls that ended in a model refusal"),
            "fallbacks": meter.create_counter(
                "rag.generation.fallbacks", unit="{fallback}",
                description="Provider calls served by a fallback model"),
            "errors": meter.create_counter(
                "rag.generation.errors", unit="{error}",
                description="Provider calls that raised an exception"),
        }
    except Exception as exc:  # noqa: BLE001 - optional telemetry must not break a question
        _trace_logger.warning("Metrics disabled (%s); question processing continues.", type(exc).__name__)
        return None


def _record(name: str, value: float, attributes: dict) -> None:
    """Add a measurement; a no-op when metrics are off and never raises."""
    try:
        instruments = _instruments()
        if instruments is None:
            return
        attributes = {k: v for k, v in attributes.items() if v is not None}
        instrument = instruments[name]
        if hasattr(instrument, "record"):
            instrument.record(value, attributes)
        else:
            instrument.add(value, attributes)
    except Exception:  # noqa: BLE001 - a telemetry failure cannot change application behavior
        _trace_logger.warning("Could not record metric.")


# Dropped-chunk count for the retrieval in progress; the ranking helpers add to it.
_retrieval_dropped = ContextVar("rag_retrieval_dropped", default=None)


def _note_dropped(count: int) -> None:
    holder = _retrieval_dropped.get()
    if holder is not None:
        holder[0] += count


@contextmanager
def _retrieval_metrics(mode: str, max_distance: float | None):
    """Time one retrieval and count its chunks; call the yielded function with the result."""
    started = perf_counter()
    holder = [0]
    token = _retrieval_dropped.set(holder)
    result: list[list[dict]] = []
    try:
        yield result.append
    except BaseException as exc:
        _record("retrieval_duration", perf_counter() - started,
                {"rag.retrieval.mode": mode, "error.type": type(exc).__name__})
        raise
    else:
        chunks = result[0] if result else []
        _record("retrieval_duration", perf_counter() - started,
                {"rag.retrieval.mode": mode, "rag.outcome": "matched" if chunks else "no_match"})
        counts = Counter(
            "section_expanded" if "expanded_from" in c
            else "kept_exact_term" if kept_by_exact_term(c, max_distance)
            else "kept"
            for c in chunks
        )
        counts["dropped"] = holder[0]
        for disposition, count in counts.items():
            if count:
                _record("chunks", count, {"rag.retrieval.mode": mode,
                                          "rag.chunk.disposition": disposition})
    finally:
        _retrieval_dropped.reset(token)


@contextmanager
def _generation_metrics(meta: dict):
    """Time one provider call and record its tokens, refusal, fallback or error from meta."""
    started = perf_counter()
    error = None
    try:
        yield
    except BaseException as exc:
        error = type(exc).__name__
        raise
    finally:
        base = {"gen_ai.operation.name": "chat",
                "gen_ai.provider.name": meta.get("provider"),
                "gen_ai.request.model": meta.get("requested_model"),
                "gen_ai.response.model": meta.get("response_model")}
        _record("operation_duration", perf_counter() - started, {**base, "error.type": error})
        if error is not None:
            _record("errors", 1, {**base, "error.type": error})
        else:
            usage = meta.get("usage") or {}
            for token_type in ("input", "output"):
                count = usage.get(f"{token_type}_tokens")
                if isinstance(count, int):
                    _record("token_usage", count, {**base, "gen_ai.token.type": token_type})
            if meta.get("stop_reason") == "refusal":
                _record("refusals", 1, base)
            if meta.get("fallback_used"):
                _record("fallbacks", 1, base)

# ── Defaults ───────────────────────────────────────────────────────────

PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "observability-docs"
N_RESULTS = 6
# Cosine distance above which a chunk is dropped. On the sample corpus the
# answerable eval questions find a relevant chunk within 0.57, and unanswerable
# questions have nothing closer than 0.65. Cosine distance tops out at 2, so
# --max-distance 2 turns the cutoff off.
MAX_DISTANCE = 0.6
# Short questions about a named thing ("What does the Ops Agent do?",
# "What is last_updated used for?") embed poorly: the right chunk ranks first
# at ~0.64 and the cutoff drops it. A chunk up to this much beyond the cutoff
# is kept when it contains an exact term from the question (see exact_terms).
EXACT_TERM_MARGIN = 0.1
# A section longer than the chunk size is split across chunks, and a retrieved
# chunk can be the half without the answer (issue #72: a table and the
# paragraph after it). When a kept chunk belongs to a section of at most this
# many chunks, the section's other chunks are appended to the results.
SECTION_MAX_CHUNKS = 3

# "vector" ranks chunks by embedding similarity alone. "hybrid" fuses that
# ranking with a BM25 keyword ranking (reciprocal rank fusion), which helps
# with exact terms such as "0.47%", "Ops Agent" or tag names that MiniLM
# embeddings blur. The distance cutoff applies to both.
RETRIEVAL_MODES = ("vector", "hybrid")
DEFAULT_RETRIEVAL = "vector"
RRF_K = 60  # standard reciprocal rank fusion constant
BM25_K1 = 1.5
BM25_B = 0.75
# Keeps decimals, percentages and hyphen/underscore names as single tokens:
# "0.47%", "cost-centre", "last_updated".
TOKEN_RE = re.compile(r"\w+(?:[.\-]\w+)*%?")


def tokenize(text: str) -> list[str]:
    """Lowercased keyword tokens for BM25."""
    return TOKEN_RE.findall(text.lower())


# Names written as a run of capitalised words: "Ops Agent", "Basic Logs".
PROPER_PHRASE_RE = re.compile(r"\b[A-Z][\w-]*(?:\s+[A-Z][\w-]*)+")


def exact_terms(query: str) -> list[str]:
    """Distinctive terms in a question that a chunk can match verbatim.

    Identifiers ("last_updated", "cost-centre", "0.47%", "p95") and multi-word
    names ("Ops Agent"). Single capitalised words such as "GCP" are too common
    in the corpus to count, and the question's first word is ignored so that
    "Which Log Analytics" reduces to "Log Analytics".
    """
    terms = [t for t in TOKEN_RE.findall(query)
             if len(t) >= 3 and (re.search(r"[_.%-]", t)
                                 or (re.search(r"\d", t) and re.search(r"[a-zA-Z]", t)))]
    words = query.split(maxsplit=1)
    rest = words[1] if len(words) > 1 else ""
    terms += [" ".join(m.split()) for m in PROPER_PHRASE_RE.findall(rest)]
    return list(dict.fromkeys(terms))


def matched_exact_terms(terms: list[str], text: str) -> list[str]:
    """The terms that appear in text as whole words, ignoring case and spacing."""
    flat = " ".join(text.split())
    return [t for t in terms
            if re.search(r"(?<!\w)" + re.escape(t) + r"(?!\w)", flat, re.IGNORECASE)]


def within_cutoff(distance: float | None, max_distance: float | None,
                  terms: list[str], text: str) -> bool:
    """The distance cutoff, relaxed by EXACT_TERM_MARGIN for exact-term matches."""
    if max_distance is None:
        return True
    if distance is None:
        return False
    if distance <= max_distance:
        return True
    return (distance <= max_distance + EXACT_TERM_MARGIN
            and bool(matched_exact_terms(terms, text)))


def kept_by_exact_term(chunk: dict, max_distance: float | None) -> bool:
    """Whether a retrieved chunk is beyond max_distance and was kept only by
    the exact-term margin (not appended by split-section completion)."""
    distance = chunk.get("distance")
    return (max_distance is not None and distance is not None
            and distance > max_distance and "expanded_from" not in chunk)


def chunk_index(chunk_id: str) -> int:
    """Position of a "<source>#<index>" chunk within its document."""
    try:
        return int(chunk_id.rsplit("#", 1)[1])
    except (IndexError, ValueError):
        return 0


def cosine_distance(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = (math.sqrt(sum(x * x for x in a)) or 1.0) * (math.sqrt(sum(y * y for y in b)) or 1.0)
    return 1 - dot / norm


def bm25_scores(query: str, documents: list[list[str]]) -> list[float]:
    """Okapi BM25 score of each tokenized document for the query."""
    if not documents:
        return []
    n_docs = len(documents)
    avg_len = sum(len(d) for d in documents) / n_docs or 1.0
    doc_freq = Counter(term for d in documents for term in set(d))
    query_terms = set(tokenize(query))
    scores = []
    for doc in documents:
        counts = Counter(doc)
        score = 0.0
        for term in query_terms:
            tf = counts.get(term, 0)
            if not tf:
                continue
            df = doc_freq[term]
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            norm = BM25_K1 * (1 - BM25_B + BM25_B * len(doc) / avg_len)
            score += idf * tf * (BM25_K1 + 1) / (tf + norm)
        scores.append(score)
    return scores


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = RRF_K) -> list[int]:
    """Fuse several rankings of the same items (best first) into one."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            fused[item] = fused.get(item, 0.0) + 1 / (k + rank)
    # Ties keep the order of the first ranking (the vector ranking).
    first = {item: i for i, item in enumerate(rankings[0])} if rankings else {}
    return sorted(fused, key=lambda item: (-fused[item], first.get(item, len(first))))

DEFAULT_PROVIDER = "anthropic"
DEFAULT_MODEL = "claude-sonnet-5-5"

PROVIDER_DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-5-5",
    "deepseek": "deepseek-chat",
}

SYSTEM_PROMPT = """You are an observability domain expert assistant.

You answer questions based ONLY on the provided context documents.
If the context does not contain enough information to answer, say so clearly.

For every factual claim, cite the source document and section in parentheses.
Be concise and precise — this is technical infrastructure knowledge.

Citation format:
    "<claim> (source: <document>.md, section '<Section>')."

Do NOT use any external knowledge. Stick strictly to what the retrieved
chunks contain."""

# Anthropic path: chunks arrive as documents with citations enabled, so the API
# attaches structured citations and the prompt doesn't ask for written ones.
ANTHROPIC_SYSTEM_PROMPT = """You are an observability domain expert assistant.

You answer questions based ONLY on the provided documents.
If the documents do not contain enough information to answer, say so clearly.

Support every factual claim with a citation from the documents.
Be concise and precise — this is technical infrastructure knowledge.

Do NOT use any external knowledge. Stick strictly to what the documents contain."""


# ── Citation display ───────────────────────────────────────────────────

def chunk_title(chunk: dict) -> str:
    """A chunk's source plus its section, when it has one."""
    if chunk["section"]:
        return f"{chunk['source']}, section '{chunk['section']}'"
    return chunk["source"]


def citation_marker(citations: list[dict]) -> str:
    """Inline marker in the prompt-citation format, e.g.
    " (source: a.md, section 'X'; source: b.md)"; empty if there are none."""
    labels = dict.fromkeys(f"source: {chunk_title(c)}" for c in citations)
    return f" ({'; '.join(labels)})" if labels else ""


def format_citations(citations: list[dict]) -> str:
    """Numbered list of structured citations with the quoted text, for the CLI."""
    lines = []
    for i, c in enumerate(citations, start=1):
        quote = " ".join(c["cited_text"].split())
        if len(quote) > 200:
            quote = quote[:197] + "..."
        lines.append(f"[{i}] {chunk_title(c)}: \"{quote}\"")
    return "\n".join(lines)


# ── Request construction ───────────────────────────────────────────────
# Pure functions, so tests and adversarial_evals.py can check offline where
# retrieved text ends up: always inside a document block (Anthropic) or after
# "Context:" in the user message (DeepSeek), never in the system prompt.

def anthropic_content(query: str, chunks: list[dict]) -> list[dict]:
    """The user turn for Claude: one citable document block per chunk, then
    the question as the only text block."""
    documents = [
        {
            "type": "document",
            "source": {"type": "text", "media_type": "text/plain", "data": c["text"]},
            # A superseded, expired, future or conflicting policy is labelled
            # in the title so the model sees it (issue #78).
            "title": chunk_title(c) + policy.note(c.get("policy")),
            "citations": {"enabled": True},
        }
        for c in chunks
    ]
    question = {
        "type": "text",
        "text": f"Answer this question using ONLY the documents above.\n\nQuestion: {query}",
    }
    return [*documents, question]


def deepseek_messages(query: str, chunks: list[dict]) -> list[dict]:
    """System and user messages for DeepSeek: the question, then the chunks as
    a prompt-written context with a "[Chunk N] Source: …" header each."""
    context = "\n\n---\n\n".join(
        f"[Chunk {i}] Source: {c['source']}"
        + (f", Section: {c['section']}" if c["section"] else "")
        + policy.note(c.get("policy"))
        + f"\n{c['text']}"
        for i, c in enumerate(chunks, start=1)
    )
    user_content = f"""Answer the following question using ONLY the context below.
Cite the source document and section for each claim you make.

Question: {query}

Context:
{context}"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# ── Generation metadata ────────────────────────────────────────────────

USAGE_FIELDS = ("input_tokens", "output_tokens",
                "cache_creation_input_tokens", "cache_read_input_tokens")


def anthropic_usage(response) -> tuple[dict | None, bool | None]:
    """Token usage of a Messages API response, and whether a server-side
    fallback model served it (None when the response doesn't say)."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return None, None
    counts = {f: getattr(usage, f, None) for f in USAGE_FIELDS}
    counts = {f: v for f, v in counts.items() if isinstance(v, int)}
    iterations = getattr(usage, "iterations", None)
    fallback = (any(getattr(i, "type", None) == "fallback_message" for i in iterations)
                if isinstance(iterations, list) else None)
    return counts or None, fallback


def openai_usage(response) -> dict | None:
    """Token usage of an OpenAI-compatible chat completion, in Anthropic's field names."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    counts = {"input_tokens": getattr(usage, "prompt_tokens", None),
              "output_tokens": getattr(usage, "completion_tokens", None)}
    counts = {f: v for f, v in counts.items() if isinstance(v, int)}
    return counts or None


# ── RAG engine ─────────────────────────────────────────────────────────

class RAGEngine:
    """Retrieve-and-generate over observability docs."""

    # What the most recent answer()/answer_with_citations() call sent to the
    # provider and got back: provider, requested and returned model IDs, stop
    # reason and token usage (None where the provider didn't report them).
    # evals.py reads it for its run artifact; the return shapes are unchanged.
    last_generation: dict | None = None

    def __init__(self, persist_dir: str = PERSIST_DIR, retrieval: str = DEFAULT_RETRIEVAL,
                 embedder: SentenceTransformer | None = None):
        """Open the active collection; pass an already-loaded embedder to reuse it
        (the HTTP service keeps one model across index reloads)."""
        if retrieval not in RETRIEVAL_MODES:
            raise ValueError(
                f"Unknown retrieval mode {retrieval!r}; expected one of {list(RETRIEVAL_MODES)}"
            )
        self.retrieval = retrieval
        self._corpus = None  # loaded on first hybrid query
        self.client = chromadb.PersistentClient(path=persist_dir)
        try:
            name = (Path(persist_dir) / "active-collection").read_text(encoding="utf-8")
        except FileNotFoundError:
            name = COLLECTION_NAME  # indexes created before replacement switching
        self.collection = self.client.get_collection(name)
        self.collection_name = name
        self.embedder = embedder if embedder is not None else SentenceTransformer(EMBED_MODEL)

    @trace_span("rag.retrieve")
    def retrieve(self, query: str, n_results: int = N_RESULTS,
                 max_distance: float | None = MAX_DISTANCE,
                 as_of: date | None = None) -> list[dict]:
        """Embed the query and retrieve the top-k matching chunks.

        Chunks with a cosine distance above max_distance (i.e. less similar)
        are dropped, unless they are within EXACT_TERM_MARGIN of it and contain
        an exact term from the query; pass None to keep all top-k.

        Each chunk carries "policy": its document's policy status on as_of
        (default: today, UTC). Applicability is not used to filter or rank:
        superseded and conflicting guidance is labelled, not hidden.
        """
        trace_attributes({"rag.retrieval.mode": self.retrieval, "rag.retrieval.k": n_results,
                          "rag.retrieval.max_distance": max_distance})
        with _retrieval_metrics(self.retrieval, max_distance) as record:
            query_embedding = self.embedder.encode([query]).tolist()[0]
            if self.retrieval == "hybrid":
                chunks = self._retrieve_hybrid(query, query_embedding, n_results, max_distance)
            else:
                chunks = self._retrieve_vector(query, query_embedding, n_results, max_distance)
            chunks = self._complete_sections(query_embedding, chunks)
            statuses = self.policy_statuses(as_of)
            day = as_of or datetime.now(UTC).date()
            for chunk in chunks:
                chunk["policy"] = (statuses or {}).get(chunk["source"]) or policy.unavailable(day)
            trace_attributes({"rag.retrieval.hit_count": len(chunks),
                              "rag.outcome": "matched" if chunks else "no_match"})
            for rank, chunk in enumerate(chunks, start=1):
                trace_attributes({f"rag.retrieval.hit.{rank}.source": chunk["source"],
                                  f"rag.retrieval.hit.{rank}.distance": chunk["distance"]})
            record(chunks)
            return chunks

    def policy_statuses(self, as_of: date | None = None) -> dict[str, dict] | None:
        """Every indexed document's policy status on as_of, or None when the
        index has no policy registry (built before issue #78)."""
        as_of = as_of or datetime.now(UTC).date()
        key = (getattr(self, "collection_name", None), as_of)
        cache = self.__dict__.setdefault("_policy_cache", {})
        if key not in cache:
            try:
                manifest = json.loads((self.collection.metadata or {}).get("corpus_manifest", "null"))
                registry = manifest.get("policies") if isinstance(manifest, dict) else None
            except (TypeError, ValueError, AttributeError):
                registry = None
            cache[key] = None if registry is None else policy.applicability(registry, as_of)
        return cache[key]

    def _retrieve_vector(self, query: str, query_embedding: list[float], n_results: int,
                         max_distance: float | None) -> list[dict]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
        )

        chunks = []
        for i in range(len(results["ids"][0])):
            meta = results["metadatas"][0][i]
            chunks.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "source": meta.get("source", "unknown"),
                "file": meta.get("file", ""),
                "section": meta.get("section", ""),
                "distance": results["distances"][0][i] if "distances" in results else None,
            })
        terms = exact_terms(query)
        kept = [c for c in chunks
                if within_cutoff(c["distance"], max_distance, terms, c["text"])]
        _note_dropped(len(chunks) - len(kept))
        return kept

    def _complete_sections(self, query_embedding: list[float],
                           chunks: list[dict]) -> list[dict]:
        """Append the other chunks of each split section a kept chunk belongs to.

        Added chunks follow every ranked chunk, in document order, so ranks and
        n_results are unchanged. Each carries its own cosine distance (it may be
        beyond the cutoff: it is kept because its section was) and
        "expanded_from", the ID of the ranked chunk that pulled it in. Chunks
        without a section and sections longer than SECTION_MAX_CHUNKS are left
        as they are.
        """
        have = {c["id"] for c in chunks}
        added: list[dict] = []
        for chunk in list(chunks):
            if not chunk.get("section"):
                continue
            siblings = self.collection.get(
                where={"$and": [{"source": chunk["source"]}, {"section": chunk["section"]}]},
                include=["documents", "metadatas", "embeddings"],
            )
            if not 1 < len(siblings["ids"]) <= SECTION_MAX_CHUNKS:
                continue
            rows = sorted(zip(siblings["ids"], siblings["documents"], siblings["metadatas"],
                              siblings["embeddings"], strict=True),
                          key=lambda row: chunk_index(row[0]))
            for chunk_id, text, meta, embedding in rows:
                if chunk_id in have:
                    continue
                have.add(chunk_id)
                added.append({
                    "id": chunk_id,
                    "text": text,
                    "source": meta.get("source", "unknown"),
                    "file": meta.get("file", ""),
                    "section": meta.get("section", ""),
                    "distance": cosine_distance(query_embedding, list(map(float, embedding))),
                    "expanded_from": chunk["id"],
                })
        return chunks + added

    def _load_corpus(self) -> dict:
        """Every chunk with its embedding and BM25 tokens, cached per engine."""
        if self._corpus is None:
            data = self.collection.get(include=["documents", "metadatas", "embeddings"])
            self._corpus = {
                "ids": list(data["ids"]),
                "documents": list(data["documents"]),
                "metadatas": list(data["metadatas"]),
                "embeddings": [list(map(float, e)) for e in data["embeddings"]],
                "tokens": [tokenize(d) for d in data["documents"]],
            }
        return self._corpus

    def _retrieve_hybrid(self, query: str, query_embedding: list[float], n_results: int,
                         max_distance: float | None) -> list[dict]:
        """Fuse cosine and BM25 rankings over the whole collection.

        The corpus is scored in memory, which suits local document sets of a
        few thousand chunks. The distance cutoff (with its exact-term margin)
        is applied after fusion, so it still decides whether a question is
        answerable.
        """
        corpus = self._load_corpus()
        q_norm = math.sqrt(sum(x * x for x in query_embedding)) or 1.0
        distances = []
        for emb in corpus["embeddings"]:
            dot = sum(a * b for a, b in zip(query_embedding, emb, strict=True))
            norm = math.sqrt(sum(x * x for x in emb)) or 1.0
            distances.append(1 - dot / (q_norm * norm))
        indices = range(len(distances))
        vector_ranking = sorted(indices, key=lambda i: distances[i])
        keyword = bm25_scores(query, corpus["tokens"])
        # Only chunks sharing a query term get a keyword rank.
        keyword_ranking = sorted((i for i in indices if keyword[i] > 0),
                                 key=lambda i: -keyword[i])

        terms = exact_terms(query)
        chunks = []
        for position, i in enumerate(reciprocal_rank_fusion([vector_ranking, keyword_ranking])):
            if not within_cutoff(distances[i], max_distance, terms, corpus["documents"][i]):
                if position < n_results:  # count drops among the top k, as in vector mode
                    _note_dropped(1)
                continue
            meta = corpus["metadatas"][i] or {}
            chunks.append({
                "id": corpus["ids"][i],
                "text": corpus["documents"][i],
                "source": meta.get("source", "unknown"),
                "file": meta.get("file", ""),
                "section": meta.get("section", ""),
                "distance": distances[i],
            })
            if len(chunks) == n_results:
                break
        return chunks

    def answer(self, query: str, n_results: int = N_RESULTS,
               verbose: bool = False,
               max_distance: float | None = MAX_DISTANCE) -> tuple[str, list[dict]]:
        """Retrieve relevant chunks and generate an answer via the configured LLM."""
        answer, chunks, _ = self.answer_with_citations(
            query, n_results=n_results, verbose=verbose, max_distance=max_distance,
        )
        return answer, chunks

    @trace_span("rag.question")
    def answer_with_citations(
        self, query: str, n_results: int = N_RESULTS, verbose: bool = False,
        max_distance: float | None = MAX_DISTANCE,
    ) -> tuple[str, list[dict], list[dict] | None]:
        """Like answer(), plus the answer's structured citations.

        Each citation is a dict with the cited chunk's "source", "section" and
        "chunk" (1-based rank) and the exact "cited_text". Citations are None
        when the provider has no native citations (DeepSeek); the answer text
        then carries the citations the model wrote.
        """
        mode = getattr(self, "retrieval", None)
        try:
            result = self._answer_with_citations(query, n_results, verbose, max_distance)
        except BaseException as exc:
            _record("questions", 1, {"rag.retrieval.mode": mode, "rag.outcome": "error",
                                     "error.type": type(exc).__name__})
            raise
        outcome = "generated" if result[1] else "no_match"  # no chunks: no provider call
        _record("questions", 1, {"rag.retrieval.mode": mode, "rag.outcome": outcome})
        return result

    def _answer_with_citations(self, query: str, n_results: int, verbose: bool,
                               max_distance: float | None):
        """answer_with_citations() without the question counter."""
        self.last_generation = None
        chunks = self.retrieve(query, n_results=n_results, max_distance=max_distance)

        if verbose:
            manifest = json.loads((self.collection.metadata or {}).get("corpus_manifest", "null"))
            if manifest:
                print(f"Indexed at: {manifest['ingested_at']}")
                print(f"Corpus SHA-256: {manifest['corpus_sha256']}")
                print(f"Embedding model: {manifest['embedding_model']}")
            else:
                print("Corpus provenance unavailable; re-ingest to record it.")

        if not chunks:
            self.last_generation = {"llm_called": False,
                                    "reason": "no chunks retrieved, so no provider call"}
            trace_attributes({"rag.outcome": "no_match", "rag.generation.called": False})
            return "No relevant documents found.", chunks, []

        if verbose:
            print(f"\n{'='*60}")
            print(f"RETRIEVED CHUNKS ({len(chunks)}):")
            print(f"{'='*60}")
            for i, c in enumerate(chunks):
                dist = f"{c['distance']:.4f}" if c["distance"] is not None else "n/a"
                print(f"\n--- Chunk {i+1} (source: {c['source']}, dist: {dist}) ---")
                print(c["text"][:400])
                if len(c["text"]) > 400:
                    print("... (truncated)")

        trace_attributes({"rag.generation.called": True})
        answer, citations = self._call_llm(query, chunks)
        trace_attributes({"rag.outcome": "generated"})
        return answer, chunks, citations

    @trace_span("rag.generate")
    def _call_llm(self, query: str, chunks: list[dict]) -> tuple[str, list[dict] | None]:
        """Send the question and chunks to the configured LLM provider.

        Returns the answer and its structured citations (None when the
        provider has no native citations).
        """
        provider = os.environ.get("LLM_PROVIDER", DEFAULT_PROVIDER)
        if provider not in PROVIDER_DEFAULT_MODELS:
            raise ValueError(
                f"Unknown LLM provider {provider!r}; "
                f"expected one of {sorted(PROVIDER_DEFAULT_MODELS)}"
            )
        model = os.environ.get("LLM_MODEL", "")
        if not model:
            model = PROVIDER_DEFAULT_MODELS.get(provider, DEFAULT_MODEL)

        meta = {"llm_called": True, "provider": provider, "requested_model": model,
                "response_model": None, "stop_reason": None, "usage": None,
                "fallback_used": None}
        self.last_generation = meta
        trace_generation(meta)
        with _generation_metrics(meta):
            return self._call_provider(provider, model, query, chunks, meta)

    def _call_provider(self, provider: str, model: str, query: str, chunks: list[dict],
                       meta: dict) -> tuple[str, list[dict] | None]:
        """The provider request itself; fills meta with what the provider reported."""
        if provider == "anthropic":
            try:
                return self._call_anthropic(query, chunks, model, meta)
            finally:
                trace_generation(meta)

        # provider == "deepseek" (OpenAI-compatible): the prompt asks for
        # citations written into the answer text.
        from openai import OpenAI
        client = OpenAI(
            base_url="https://api.deepseek.com",
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
        )
        response = client.chat.completions.create(
            model=model,
            max_tokens=1024,
            messages=deepseek_messages(query, chunks),
        )
        returned = getattr(response, "model", None)
        meta["response_model"] = returned if isinstance(returned, str) else None
        finish = getattr(response.choices[0], "finish_reason", None)
        meta["stop_reason"] = finish if isinstance(finish, str) else None
        meta["usage"] = openai_usage(response)
        trace_generation(meta)
        return response.choices[0].message.content, None

    @staticmethod
    def _call_anthropic(query: str, chunks: list[dict], model: str,
                        meta: dict | None = None) -> tuple[str, list[dict]]:
        """Ask Claude with each chunk as a document with citations enabled.

        Each cited span of the answer is followed by a "(source: …, section
        '…')" marker, so it displays like a prompt-cited answer. When meta is
        given, the returned model ID, stop reason and token usage are recorded
        in it.
        """
        import anthropic
        client = anthropic.Anthropic()
        # Sonnet 5.5 always thinks, and thinking counts against max_tokens,
        # so leave headroom. On a safety refusal, the server retries on a
        # fallback model ("default" routing by refusal category).
        response = client.beta.messages.create(
            model=model,
            max_tokens=16000,
            system=ANTHROPIC_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": anthropic_content(query, chunks)}],
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if meta is not None:
            # The model that served the answer: a fallback model when the
            # requested one declined.
            returned = getattr(response, "model", None)
            meta["response_model"] = returned if isinstance(returned, str) else None
            meta["stop_reason"] = response.stop_reason
            meta["usage"], meta["fallback_used"] = anthropic_usage(response)
        if response.stop_reason == "refusal":
            return "The model declined to answer this question.", []

        # content may start with thinking blocks; the answer arrives split into
        # text blocks, and each cited block carries its own citations.
        parts: list[str] = []
        citations: list[dict] = []
        for block in response.content:
            if block.type != "text":
                continue
            block_citations = []
            for cite in getattr(block, "citations", None) or []:
                index = getattr(cite, "document_index", None)
                if index is None or not 0 <= index < len(chunks):
                    continue
                block_citations.append({
                    "source": chunks[index]["source"],
                    "section": chunks[index]["section"],
                    "chunk": index + 1,
                    "cited_text": cite.cited_text,
                })
            parts.append(block.text + citation_marker(block_citations))
            citations.extend(block_citations)
        return "".join(parts), citations


# ── CLI ────────────────────────────────────────────────────────────────

def print_answer(answer: str, citations: list[dict] | None, prefix: str) -> None:
    """Print the answer, then the quoted text behind each native citation."""
    print(f"{prefix}{answer}")
    if citations:
        print(f"\nCitations:\n{format_citations(citations)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query the observability RAG system",
    )
    parser.add_argument("query", nargs="?", help="Question to answer")
    parser.add_argument("--n-results", type=int, default=N_RESULTS,
                        help=f"Number of chunks to retrieve (default: {N_RESULTS})")
    parser.add_argument("--persist-dir", default=PERSIST_DIR)
    parser.add_argument("--max-distance", type=float, default=MAX_DISTANCE,
                        help="Drop chunks with cosine distance above this "
                             f"(default: {MAX_DISTANCE}; 2 disables the cutoff)")
    parser.add_argument("--retrieval", default=DEFAULT_RETRIEVAL, choices=RETRIEVAL_MODES,
                        help="vector: embedding similarity only; hybrid: fuse it with "
                             f"BM25 keyword ranking (default: {DEFAULT_RETRIEVAL})")
    parser.add_argument("--provider", default=None,
                        choices=sorted(PROVIDER_DEFAULT_MODELS),
                        help="LLM provider. Overrides LLM_PROVIDER env var.")
    parser.add_argument("--model", default=None,
                        help="Model name. Overrides LLM_MODEL env var.")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show retrieved chunks before the answer")
    args = parser.parse_args()

    # Allow CLI flags to override env vars
    if args.provider:
        os.environ["LLM_PROVIDER"] = args.provider
    if args.model:
        os.environ["LLM_MODEL"] = args.model

    provider = os.environ.get("LLM_PROVIDER", DEFAULT_PROVIDER)
    if provider not in PROVIDER_DEFAULT_MODELS:
        parser.error(f"LLM_PROVIDER={provider!r} is not one of {sorted(PROVIDER_DEFAULT_MODELS)}")

    engine = RAGEngine(persist_dir=args.persist_dir, retrieval=args.retrieval)

    if args.query:
        answer, _, citations = engine.answer_with_citations(
            args.query,
            n_results=args.n_results,
            verbose=args.verbose,
            max_distance=args.max_distance,
        )
        print_answer(answer, citations, "\nAnswer:\n")
    else:
        # Interactive REPL
        model = os.environ.get("LLM_MODEL", "") or PROVIDER_DEFAULT_MODELS.get(provider, DEFAULT_MODEL)
        print(f"RAG observability query — Provider: {provider}, Model: {model} — type 'quit' to exit.\n")
        while True:
            try:
                q = input("Q: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if q.lower() in ("quit", "exit", "q"):
                break
            if not q:
                continue
            answer, _, citations = engine.answer_with_citations(
                q, n_results=args.n_results, verbose=args.verbose,
                max_distance=args.max_distance,
            )
            print_answer(answer, citations, "A: ")
            print()


if __name__ == "__main__":
    main()
