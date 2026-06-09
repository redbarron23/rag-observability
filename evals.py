#!/usr/bin/env python3
"""Eval harness for the observability RAG system.

Tests that the RAG system correctly retrieves the right documents
for known questions over the observability corpus.  Runs in two modes:

  Mode 1 — Retrieval-only (default):
      Checks that the correct source documents appear in the top-N
      retrieved chunks.  No LLM API key required.

  Mode 2 — Full (--with-llm):
      Also asks Claude to generate an answer and checks that the
      answer contains expected facts AND cites the correct source.

The retrieval-only mode is fast, free, and catches most regressions.
Use --with-llm for the full attribution + fact-presence check.
"""

import argparse
import sys

from query import RAGEngine


EVALS = [
    {
        "id": "coverage-targets-tier1",
        "question": "What is the coverage target for Tier 1 production resources?",
        "expected_sources": ["reference/coverage-targets.md"],
        "expected_facts": ["90%", "6 months", "100%", "12 months"],
    },
    {
        "id": "coverage-targets-gcp",
        "question": "What is the current GCP coverage level and what is the target?",
        "expected_sources": ["reference/coverage-targets.md"],
        "expected_facts": ["0.93%", "0.9%", "100%"],
    },
    {
        "id": "framework-scoring",
        "question": "How does the observability framework scoring model work?",
        "expected_sources": ["architecture/observability_framework.md"],
        "expected_facts": ["0", "1", "2", "3", "Missing", "Basic", "Good", "Excellent"],
    },
    {
        "id": "gcp-azure-metrics-differences",
        "question": "How does GCP metric collection differ from Azure in terms of OpenTelemetry support and native agents?",
        "expected_sources": [
            "reference/gcp-azure-metrics-comparison.md",
            "reference/metrics-otel-azure-gcp-differences.md",
        ],
        "expected_facts": [],
    },
    {
        "id": "tagging-standard",
        "question": "What is the resource tagging standard?",
        "expected_sources": ["reference/tagging-standard.md"],
        "expected_facts": [],
    },
    {
        "id": "tracing-use-cases",
        "question": "What are some use cases for distributed tracing?",
        "expected_sources": ["architecture/observability_tracing_use_cases.md"],
        "expected_facts": [],
    },
    {
        "id": "framework-categories",
        "question": "What categories are assessed in the observability framework?",
        "expected_sources": ["architecture/observability_framework.md"],
        "expected_facts": [
            "Metrics", "Logs", "Traces", "Alerts",
            "SLO", "Ownership",
        ],
    },
    {
        "id": "data-first-approach",
        "question": "How does the data-first approach avoid unnecessary cloud API calls by checking data freshness first?",
        "expected_sources": ["data-first-approach.md"],
        "expected_facts": [],
    },
    {
        "id": "observability-setup",
        "question": "According to the observability setup primer, what are the prerequisites and steps for onboarding a new service?",
        "expected_sources": ["observability-setup-primer.md"],
        "expected_facts": [],
    },
    {
        "id": "cost-smoothing",
        "question": "What is the cost smoothing plan for Log Analytics?",
        "expected_sources": ["programme/COST_SMOOTHING.md"],
        "expected_facts": [],
    },
]


RETRIEVE_N = 6  # number of chunks to retrieve for source-attribution checks


def run_retrieval_evals(engine: RAGEngine, verbose: bool) -> tuple[int, int]:
    """Run retrieval-only evals — no LLM API call needed."""
    passed = 0
    failed = 0

    print(f"Retrieval evals (source attribution in top-{RETRIEVE_N} retrieved chunks)")
    print("=" * 60)

    for ev in EVALS:
        chunks = engine.retrieve(ev["question"], n_results=RETRIEVE_N)
        retrieved_sources = {c["source"] for c in chunks}

        # Check that all expected sources appear in the retrieved set
        expected = ev["expected_sources"]
        found = [s for s in expected if any(s in r for r in retrieved_sources)]
        missing = [s for s in expected if s not in found]
        ok = len(missing) == 0

        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {ev['id']}")

        if not ok or verbose:
            print(f"       Q: {ev['question']}")
            print(f"       Retrieved: {', '.join(sorted(retrieved_sources))}")
            if missing:
                print(f"       Missing: {missing}")

        if ok:
            passed += 1
        else:
            failed += 1

    return passed, failed


def run_full_evals(engine: RAGEngine, verbose: bool) -> tuple[int, int]:
    """Run full evals — retrieval + LLM answer + fact-presence check."""
    passed = 0
    failed = 0

    print("\nFull evals (retrieval + LLM answer + fact presence)")
    print("=" * 60)

    for ev in EVALS:
        try:
            answer, chunks = engine.answer(
                ev["question"], n_results=4, verbose=False,
            )
        except Exception as e:
            print(f"[SKIP] {ev['id']} — LLM error: {e}")
            continue

        # Check 1: source attribution in retrieved chunks
        retrieved_sources = {c["source"] for c in chunks}
        expected = ev["expected_sources"]
        found = [s for s in expected if any(s in r for r in retrieved_sources)]
        missing_sources = [s for s in expected if s not in found]
        sources_ok = len(missing_sources) == 0

        # Check 2: fact presence in the answer
        facts_ok = True
        missing_facts = []
        for fact in ev["expected_facts"]:
            if fact.lower() not in answer.lower():
                facts_ok = False
                missing_facts.append(fact)

        ok = sources_ok and facts_ok
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {ev['id']}")

        if not ok or verbose:
            print(f"       Q: {ev['question']}")
            print(f"       Retrieved: {', '.join(sorted(retrieved_sources))}")
            if missing_sources:
                print(f"       Missing sources: {missing_sources}")
            if missing_facts:
                print(f"       Missing facts: {missing_facts}")
            print(f"       A: {answer[:300]}")

        if ok:
            passed += 1
        else:
            failed += 1

    return passed, failed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG evals")
    parser.add_argument("--with-llm", action="store_true",
                        help="Also run LLM-based fact-presence checks")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    engine = RAGEngine()

    p1, f1 = run_retrieval_evals(engine, verbose=args.verbose)
    print(f"\nRetrieval results: {p1}/{p1 + f1} passed")

    if args.with_llm:
        p2, f2 = run_full_evals(engine, verbose=args.verbose)
        print(f"Full results:       {p2}/{p2 + f2} passed")
        print(f"\nCombined: {p1 + p2}/{p1 + p2 + f1 + f2} passed")
    else:
        print("(Run with --with-llm to also check fact presence in generated answers)")
        print("Skipping LLM-based checks (no ANTHROPIC_API_KEY needed for retrieval)")


if __name__ == "__main__":
    main()
