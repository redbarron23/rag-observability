"""Machine-readable eval run artifacts (issue #39).

`python evals.py --output run.json` writes one JSON document describing what
was evaluated (code revision, indexed corpus, embedding model, retrieval
settings, generation provider and model) and what happened (per-case outcomes,
dimension results, provider errors, timings and token usage). Two artifacts
can then be compared case by case; see "Comparing eval runs" in docs/guide.md.

Rules the artifact follows:

- Metadata that can't be determined is null with a reason next to it
  (``*_unavailable_reason``); nothing is guessed or filled in from defaults.
- The model ID is the one the provider returned, alongside the requested one.
- No API keys: provider error messages and answers are scrubbed of the values
  of the key environment variables before they are written.
- No document text, and no answers unless ``--include-answers`` is given.
  Problem messages can quote the failing phrase of an answer (for example the
  unsupported sentence after a decline), as the console output does.
- Held-out results sit under their own top-level key and never count towards
  the exit code.
"""

from __future__ import annotations

import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 1
ARTIFACT_KIND = "rag-observability-eval-run"
SECRET_ENV_VARS = ("ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY", "RAG_API_KEY", "OPENAI_API_KEY")
REDACTED = "[REDACTED]"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def redact(text: str | None) -> str | None:
    """Replace the value of any API key environment variable found in text."""
    if text is None:
        return None
    for name in SECRET_ENV_VARS:
        value = os.environ.get(name, "")
        if len(value) >= 8:  # don't blank out common short strings
            text = text.replace(value, REDACTED)
    return text


def git_revision(cwd: str | Path | None = None) -> dict:
    """The checked-out commit and whether tracked files have local changes."""
    cwd = cwd or Path(__file__).resolve().parent
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True,
                             text=True, timeout=10, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError) as e:
        return {"git_sha": None, "git_dirty": None,
                "unavailable_reason": f"git rev-parse failed: {type(e).__name__}"}
    try:
        status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                cwd=cwd, capture_output=True, text=True, timeout=10,
                                check=True).stdout
        dirty = bool(status.strip())
    except (OSError, subprocess.SubprocessError):
        dirty = None
    return {"git_sha": sha, "git_dirty": dirty}


def corpus_metadata(engine) -> dict:
    """Corpus identity of the evaluated index (reuses api.corpus_identity)."""
    from api import corpus_identity
    try:
        identity = corpus_identity(engine)
    except Exception as e:  # noqa: BLE001 - provenance is best-effort, never fatal
        return {"provenance": "unavailable",
                "unavailable_reason": f"could not read the index: {type(e).__name__}"}
    if identity.get("provenance") != "available":
        identity["unavailable_reason"] = (
            "the collection has no supported corpus manifest; re-ingest to record it")
    return identity


def embedding_metadata(corpus: dict) -> dict:
    """The embedding model recorded with the index, and the locally installed stack."""
    from ingest import EMBED_MODEL, embedding_versions
    indexed = None
    if corpus.get("provenance") == "available":
        indexed = {"model": corpus.get("embedding_model"),
                   "revision": corpus.get("embedding_revision")}
    installed = embedding_versions()
    out = {
        "indexed": indexed,
        "installed": {"model": EMBED_MODEL,
                      "revision": installed["embedding_revision"],
                      "sentence_transformers_version":
                          installed["sentence_transformers_version"]},
    }
    if indexed is None:
        out["indexed_unavailable_reason"] = "no corpus manifest on the index"
    elif indexed["revision"] is None:
        out["indexed_revision_unavailable_reason"] = (
            "the model revision wasn't resolvable from the local cache at ingest time")
    return out


def retrieval_settings(mode: str, retrieve_n: int) -> dict:
    import query
    settings = {
        "mode": mode,
        "retrieval_check_k": retrieve_n,
        "generation_k": query.N_RESULTS,
        "max_distance": query.MAX_DISTANCE,
        "exact_term_margin": query.EXACT_TERM_MARGIN,
        "section_max_chunks": query.SECTION_MAX_CHUNKS,
        # Full evals of unanswerable questions run without the cutoff.
        "unanswerable_generation_max_distance": None,
    }
    if mode == "hybrid":
        settings.update(rrf_k=query.RRF_K, bm25_k1=query.BM25_K1, bm25_b=query.BM25_B)
    return settings


def dimension_summary(cases: list[dict]) -> dict:
    """Per dimension: passed, failed and unevaluated (None) counts."""
    dims: dict[str, dict] = {}
    for case in cases:
        for name, ok in case["dimensions"].items():
            d = dims.setdefault(name, {"passed": 0, "failed": 0, "unevaluated": 0})
            d["passed" if ok is True else "failed" if ok is False else "unevaluated"] += 1
    return dims


def section_summary(cases: list[dict], retrieve_n: int) -> dict | None:
    """Supporting-section recall at the ranks the console reports."""
    ranks = [s["rank"] for c in cases for s in c.get("section_ranks", [])]
    if not ranks:
        return None
    return {"total": len(ranks), **{
        f"at_{cutoff}": sum(r is not None and r <= cutoff for r in ranks)
        for cutoff in (1, 3, retrieve_n)}}


def usage_totals(cases: list[dict]) -> dict:
    """Summed token usage over the cases whose provider reported it."""
    totals: dict[str, int] = {}
    calls = reported = 0
    for case in cases:
        gen = case.get("generation") or {}
        if not gen.get("llm_called"):
            continue
        calls += 1
        if gen.get("usage"):
            reported += 1
            for field, value in gen["usage"].items():
                totals[field] = totals.get(field, 0) + value
    return {"provider_calls": calls, "calls_with_usage": reported, "tokens": totals}


def section(cases: list[dict], retrieve_n: int) -> dict:
    """One results block (e.g. retrieval or full) with its summary."""
    passed = sum(c["status"] == "pass" for c in cases)
    summary = {
        "passed": passed,
        "failed": len(cases) - passed,
        "total": len(cases),
        "provider_errors": sum(c["status"] == "error" for c in cases),
        "by_dimension": dimension_summary(cases),
    }
    sections = section_summary(cases, retrieve_n)
    if sections is not None:
        summary["supporting_section_recall"] = sections
    if any("exact_term_margin" in c for c in cases):
        # Cases whose supporting chunk only the exact-term margin kept; diff
        # this across runs to see corpus edits move cases in or out of it.
        summary["exact_term_margin_cases"] = [c["id"] for c in cases
                                              if c.get("exact_term_margin")]
    if any("generation" in c for c in cases):
        summary["usage"] = usage_totals(cases)
    return {"summary": summary, "cases": cases}


def generation_metadata(with_llm: bool, blocks: list[list[dict]]) -> dict:
    """Configured provider and model plus what the provider actually returned."""
    if not with_llm:
        return {"provider": None, "requested_model": None, "response_models": [],
                "unavailable_reason": "retrieval-only run (no --with-llm); no provider calls"}
    from query import DEFAULT_PROVIDER, PROVIDER_DEFAULT_MODELS
    provider = os.environ.get("LLM_PROVIDER", DEFAULT_PROVIDER)
    requested = os.environ.get("LLM_MODEL", "") or PROVIDER_DEFAULT_MODELS.get(provider)
    gens = [c["generation"] for cases in blocks for c in cases
            if (c.get("generation") or {}).get("llm_called")]
    returned = sorted({g["response_model"] for g in gens if g.get("response_model")})
    out = {"provider": provider, "requested_model": requested, "response_models": returned,
           "fallback_calls": sum(g.get("fallback_used") is True for g in gens)}
    if not returned:
        out["response_models_unavailable_reason"] = (
            "no provider response reported a model ID" if gens else "no provider calls succeeded")
    return out


def build_artifact(*, engine, mode: str, retrieve_n: int, with_llm: bool,
                   include_answers: bool, started_at: str, finished_at: str,
                   duration_seconds: float, exit_code: int,
                   results: dict[str, list[dict]]) -> dict:
    """Assemble the artifact. `results` maps "retrieval", "full",
    "heldout_retrieval" and "heldout_full" to case records (full ones absent
    without --with-llm)."""
    def block(cases):
        return section(cases, retrieve_n)

    corpus = corpus_metadata(engine)
    full = results.get("full")
    heldout_full = results.get("heldout_full")
    not_run = {"not_run_reason": "retrieval-only run (no --with-llm)"}
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": ARTIFACT_KIND,
        "run": {
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_seconds": round(duration_seconds, 3),
            "with_llm": with_llm,
            "include_answers": include_answers,
            "exit_code": exit_code,
            "passed": exit_code == 0,
        },
        "code": git_revision(),
        "corpus": corpus,
        "embedding": embedding_metadata(corpus),
        "retrieval": retrieval_settings(mode, retrieve_n),
        "generation": generation_metadata(
            with_llm, [b for b in (full, heldout_full) if b is not None]),
        "results": {
            "retrieval": block(results["retrieval"]),
            "full": block(full) if full is not None else not_run,
        },
        # Reported separately: not used for tuning, not counted in the exit code.
        "heldout": {
            "counts_toward_exit_code": False,
            "retrieval": block(results["heldout_retrieval"]),
            "full": block(heldout_full) if heldout_full is not None else not_run,
        },
    }
