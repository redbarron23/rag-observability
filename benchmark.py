#!/usr/bin/env python3
"""Benchmark keyword, vector and hybrid retrieval on the fixed eval questions.

    python ingest.py
    python benchmark.py --output docs/benchmarks/retrieval-YYYY-MM-DD.json

Issue #34: before adding retrieval complexity (#8), measure whether the
existing vector pipeline beats a plain keyword baseline on this corpus.

Three retrievers answer the same questions from the same indexed chunks:

  keyword  BM25 top-k over the chunk text (KeywordRetriever below). No
           embeddings, no distance cutoff, no split-section completion: the
           simplest thing a "search the docs" box would do.
  vector   RAGEngine(retrieval="vector"), the default pipeline.
  hybrid   RAGEngine(retrieval="hybrid"), vector + BM25 fused by RRF.

Quality is scored by evals.run_retrieval_evals, the same checks CI runs
(document attribution, supporting-section recall @1/@3/@6, fact evidence,
unanswerable rejection), on the main and held-out sets. Latency is measured
two ways:

  cold  a fresh Python process per run: import time, retriever construction
        (opening Chroma, loading the embedding model) and the first query.
        The OS file cache is not flushed, so model files are read from memory
        after the first run.
  warm  each question retrieved --repeats times in one process after the
        quality pass has warmed it; median, p95 and mean per query.

No provider is called: this is retrieval only. Generated-answer token usage
comes from `python evals.py --with-llm --output ...` and is reported
separately (docs/benchmark.md).
"""

import argparse
import contextlib
import io
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

MODES = ("keyword", "vector", "hybrid")
PERSIST_DIR = "chroma_db"


class KeywordRetriever:
    """BM25 keyword retrieval over the indexed chunks (the #34 baseline).

    Reuses query.tokenize and query.bm25_scores, the scorer hybrid mode uses,
    so the baseline differs from hybrid only by dropping the vector ranking,
    the distance cutoff and split-section completion. Chunks sharing no term
    with the question are never returned; anything else is, however weak, so
    the baseline has no way to decline an off-topic question.
    """

    retrieval = "keyword"

    def __init__(self, chunks: list[dict]):
        from query import tokenize
        self.chunks = chunks
        self.tokens = [tokenize(c["text"]) for c in chunks]

    @classmethod
    def from_index(cls, persist_dir: str = PERSIST_DIR) -> "KeywordRetriever":
        """Load every chunk of the active collection (no embeddings)."""
        from ingest import _active_collection
        collection, _ = _active_collection(persist_dir)
        data = collection.get(include=["documents", "metadatas"])
        chunks = [{"id": chunk_id, "text": text,
                   "source": (meta or {}).get("source", "unknown"),
                   "file": (meta or {}).get("file", ""),
                   "section": (meta or {}).get("section", "")}
                  for chunk_id, text, meta in zip(data["ids"], data["documents"],
                                                  data["metadatas"], strict=True)]
        retriever = cls(chunks)
        retriever.collection = collection
        return retriever

    def retrieve(self, query: str, n_results: int = 6, **_ignored) -> list[dict]:
        """Top n_results chunks by BM25 score; ties keep index order."""
        from query import bm25_scores
        scores = bm25_scores(query, self.tokens)
        ranked = sorted((i for i, s in enumerate(scores) if s > 0), key=lambda i: -scores[i])
        return [{**self.chunks[i], "score": scores[i], "distance": None}
                for i in ranked[:n_results]]


def build_retriever(mode: str, persist_dir: str, embedder=None):
    if mode == "keyword":
        return KeywordRetriever.from_index(persist_dir)
    from query import RAGEngine
    return RAGEngine(persist_dir=persist_dir, retrieval=mode, embedder=embedder)


# ── Quality ────────────────────────────────────────────────────────────

def quality(retriever) -> dict:
    """Retrieval eval summaries (main and held-out) via the CI checks."""
    import evals
    blocks = {}
    for name, answerable, unanswerable in (
        ("main", evals.EVALS, evals.UNANSWERABLE_EVALS),
        ("heldout", evals.HELDOUT_EVALS, evals.HELDOUT_UNANSWERABLE_EVALS),
    ):
        records: list[dict] = []
        with contextlib.redirect_stdout(io.StringIO()):
            evals.run_retrieval_evals(retriever, verbose=False, answerable=answerable,
                                      unanswerable=unanswerable, records=records)
        blocks[name] = summarise(records, evals.RETRIEVE_N)
    return blocks


def summarise(records: list[dict], retrieve_n: int) -> dict:
    """Pass counts per kind and dimension, section recall, and failing cases."""
    import eval_artifact
    answerable = [r for r in records if r["kind"] == "answerable"]
    unanswerable = [r for r in records if r["kind"] == "unanswerable"]
    first_ranks = [r.get("first_relevant_rank") for r in answerable]
    summary = eval_artifact.section(answerable, retrieve_n)["summary"]
    return {
        "answerable": {
            "passed": summary["passed"], "total": summary["total"],
            "by_dimension": summary["by_dimension"],
            "supporting_section_recall": summary.get("supporting_section_recall"),
            # Mean reciprocal rank of each case's first supporting section
            # (0 when missing): a single number for "how high is the evidence".
            "mrr_first_supporting_section": round(
                statistics.fmean(1 / r if r else 0.0 for r in first_ranks), 3)
            if first_ranks else None,
            "failed_cases": [r["id"] for r in answerable if r["status"] != "pass"],
        },
        "unanswerable": {
            "rejected": sum(r["status"] == "pass" for r in unanswerable),
            "total": len(unanswerable),
            "failed_cases": [r["id"] for r in unanswerable if r["status"] != "pass"],
        },
    }


# ── Latency ────────────────────────────────────────────────────────────

def questions() -> list[str]:
    import evals
    return [ev["question"] for ev in evals.EVALS + evals.UNANSWERABLE_EVALS]


def warm_latency(retriever, repeats: int) -> dict:
    """Per-query retrieval time in ms over every main-set question."""
    import evals
    samples = []
    for _ in range(repeats):
        for q in questions():
            started = time.perf_counter()
            retriever.retrieve(q, n_results=evals.RETRIEVE_N)
            samples.append((time.perf_counter() - started) * 1000)
    return latency_summary(samples)


def latency_summary(samples_ms: list[float]) -> dict:
    ordered = sorted(samples_ms)
    p95 = ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))]
    return {"queries": len(ordered), "median_ms": round(statistics.median(ordered), 2),
            "p95_ms": round(p95, 2), "mean_ms": round(statistics.fmean(ordered), 2),
            "max_ms": round(ordered[-1], 2)}


def cold_probe(mode: str, persist_dir: str) -> dict:
    """Run in a fresh process: time imports, construction and the first query."""
    t0 = time.perf_counter()
    import query  # noqa: F401 - timed: pulls in chromadb and sentence-transformers
    t1 = time.perf_counter()
    with contextlib.redirect_stderr(io.StringIO()):  # model-loading progress bars
        retriever = build_retriever(mode, persist_dir)
    t2 = time.perf_counter()
    retriever.retrieve(questions()[0], n_results=6)
    t3 = time.perf_counter()
    return {"import_s": round(t1 - t0, 3), "init_s": round(t2 - t1, 3),
            "first_query_s": round(t3 - t2, 3)}


def cold_latency(mode: str, persist_dir: str, runs: int) -> dict:
    """Median of `runs` fresh-process cold starts, plus the raw runs."""
    results = []
    for _ in range(runs):
        started = time.perf_counter()
        out = subprocess.run(
            [sys.executable, __file__, "--cold-probe", mode, "--persist-dir", persist_dir],
            capture_output=True, text=True, check=True, env={**os.environ, "TQDM_DISABLE": "1"},
        ).stdout
        probe = json.loads(out.strip().splitlines()[-1])
        probe["process_wall_s"] = round(time.perf_counter() - started, 3)
        results.append(probe)
    keys = ("import_s", "init_s", "first_query_s", "process_wall_s")
    return {"runs": results,
            "median": {k: round(statistics.median(r[k] for r in results), 3) for k in keys}}


# ── Environment ────────────────────────────────────────────────────────

def machine() -> dict:
    info = {"platform": platform.platform(), "machine": platform.machine(),
            "processor": platform.processor() or None, "cpu_count": os.cpu_count(),
            "python": platform.python_version()}
    try:
        info["memory_gb"] = round(
            os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
    except (ValueError, OSError, AttributeError):
        info["memory_gb"] = None
    if sys.platform == "darwin":
        with contextlib.suppress(OSError, subprocess.SubprocessError):
            info["cpu_model"] = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                text=True, check=True, timeout=5).stdout.strip()
    from importlib.metadata import version
    info["packages"] = {p: version(p) for p in ("chromadb", "sentence-transformers", "torch")}
    return info


def corpus_size(retriever) -> dict:
    """Documents, chunks and words in the benchmarked index."""
    chunks = retriever.chunks
    return {"documents": len({c["source"] for c in chunks}), "chunks": len(chunks),
            "words": sum(len(c["text"].split()) for c in chunks),
            "characters": sum(len(c["text"]) for c in chunks)}


def settings() -> dict:
    import evals
    import query
    return {"k": evals.RETRIEVE_N, "max_distance": query.MAX_DISTANCE,
            "exact_term_margin": query.EXACT_TERM_MARGIN,
            "section_max_chunks": query.SECTION_MAX_CHUNKS, "rrf_k": query.RRF_K,
            "bm25_k1": query.BM25_K1, "bm25_b": query.BM25_B,
            "keyword": "BM25 top-k, no distance cutoff, no section completion",
            "vector": "cosine top-k, distance cutoff + exact-term margin, section completion",
            "hybrid": "RRF(cosine, BM25) top-k, same cutoff and section completion"}


def run(persist_dir: str, repeats: int, cold_runs: int) -> dict:
    from sentence_transformers import SentenceTransformer

    import eval_artifact
    from ingest import EMBED_MODEL
    started_at = eval_artifact.utc_now()
    with contextlib.redirect_stderr(io.StringIO()):
        embedder = SentenceTransformer(EMBED_MODEL)
    keyword = KeywordRetriever.from_index(persist_dir)
    vector = build_retriever("vector", persist_dir, embedder)
    corpus = eval_artifact.corpus_metadata(vector)
    modes = {}
    for mode, retriever in (("keyword", keyword), ("vector", vector),
                            ("hybrid", build_retriever("hybrid", persist_dir, embedder))):
        print(f"{mode}: quality", file=sys.stderr)
        result = {"quality": quality(retriever)}
        print(f"{mode}: warm latency", file=sys.stderr)
        result["warm"] = warm_latency(retriever, repeats)
        print(f"{mode}: cold latency", file=sys.stderr)
        result["cold"] = cold_latency(mode, persist_dir, cold_runs)
        modes[mode] = result
    return {
        "kind": "retrieval-benchmark", "schema_version": 1, "issue": 34,
        "started_at": started_at, "finished_at": eval_artifact.utc_now(),
        "code": eval_artifact.git_revision(),
        "corpus": {**corpus, **corpus_size(keyword)},
        "embedding": eval_artifact.embedding_metadata(corpus),
        "settings": {**settings(), "warm_repeats": repeats, "cold_runs": cold_runs},
        "machine": machine(),
        "generation": {"llm_called": False,
                       "reason": "retrieval-only benchmark; see evals.py --with-llm"},
        "modes": modes,
    }


def report(result: dict) -> str:
    """A plain-text table of the headline numbers."""
    lines = [f"{'mode':8} {'pass':>7} {'docs':>7} {'@1':>7} {'@3':>7} {'@6':>7} "
             f"{'MRR':>6} {'reject':>7} {'warm p50':>9} {'warm p95':>9} {'cold':>7}"]
    for mode, r in result["modes"].items():
        a, u = r["quality"]["main"]["answerable"], r["quality"]["main"]["unanswerable"]
        rec = a["supporting_section_recall"]
        docs = a["by_dimension"]["sources"]["passed"]
        cold = r["cold"]["median"]
        lines.append(
            f"{mode:8} {a['passed']:>3}/{a['total']:<3} {docs:>3}/{a['total']:<3} "
            f"{rec['at_1']:>3}/{rec['total']:<3} {rec['at_3']:>3}/{rec['total']:<3} "
            f"{rec['at_6']:>3}/{rec['total']:<3} {a['mrr_first_supporting_section']:>6} "
            f"{u['rejected']:>3}/{u['total']:<3} {r['warm']['median_ms']:>7}ms "
            f"{r['warm']['p95_ms']:>7}ms {cold['init_s'] + cold['first_query_s']:>6.3f}s")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--persist-dir", default=PERSIST_DIR)
    parser.add_argument("--repeats", type=int, default=5,
                        help="warm passes over the question set (default 5)")
    parser.add_argument("--cold-runs", type=int, default=3,
                        help="fresh-process cold starts per mode (default 3)")
    parser.add_argument("--output", "-o", metavar="PATH", help="write the JSON results here")
    parser.add_argument("--cold-probe", choices=MODES, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.cold_probe:
        print(json.dumps(cold_probe(args.cold_probe, args.persist_dir)))
        return
    result = run(args.persist_dir, args.repeats, args.cold_runs)
    print(f"Retrieval benchmark, {datetime.now(UTC):%Y-%m-%d}, "
          f"{result['corpus']['documents']} documents / {result['corpus']['chunks']} chunks, "
          f"k={result['settings']['k']}")
    print(report(result))
    print("cold = median fresh-process construction + first query (imports excluded)")
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
