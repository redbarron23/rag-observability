#!/usr/bin/env python3
"""Eval harness for the observability RAG system.

Tests that the RAG system correctly retrieves the right documents
for known questions over the observability corpus, and declines
questions the corpus can't answer.  Runs in two modes:

  Mode 1 — Retrieval-only (default):
      Checks that the correct documents and supporting sections appear in
      the top-N retrieved chunks, and that nothing survives the distance cutoff
      for unanswerable questions.  No LLM API key required.

  Mode 2 — Full (--with-llm):
      Also asks the configured LLM to generate an answer and checks that the
      answer contains the expected facts and cites the right sources, and
      that unanswerable questions are declined.  An LLM error counts as a
      failure.

The retrieval-only mode is fast, free, and catches most regressions.
Use --with-llm for the full attribution + fact-presence + citation check.
"""

import argparse
import re
import sys

from query import DEFAULT_RETRIEVAL, N_RESULTS, RETRIEVAL_MODES, RAGEngine

EVALS = [
    {
        "id": "coverage-targets-tier1",
        "question": "What is the coverage target for Tier 1 production resources?",
        "expected_sources": ["reference/coverage-targets.md"],
        "expected_sections": [("reference/coverage-targets.md", "Tier 1")],
        "expected_facts": ["85%", "4 months", "100%", "9 months"],
    },
    {
        "id": "coverage-targets-gcp",
        "question": "What is the current GCP coverage level and what is the interim target?",
        "expected_sources": ["reference/coverage-targets.md"],
        "expected_sections": [("reference/coverage-targets.md", "Current status")],
        "expected_facts": ["0.47%", "40%"],
    },
    {
        "id": "framework-scoring",
        "question": "How does the observability framework scoring model work?",
        "expected_sources": ["architecture/observability_framework.md"],
        "expected_sections": [("architecture/observability_framework.md", "Scoring model")],
        "expected_facts": ["Missing", "Basic", "Good", "Excellent"],
    },
    {
        "id": "framework-categories",
        "question": "What categories are assessed in the observability framework?",
        "expected_sources": ["architecture/observability_framework.md"],
        "expected_sections": [("architecture/observability_framework.md", "Categories")],
        "expected_facts": ["Metrics", "Logs", "Traces", "Alerts", "SLO", "Ownership"],
    },
    {
        "id": "gcp-azure-metrics-differences",
        "question": "How does GCP metric collection differ from Azure in terms of OpenTelemetry support and native agents?",
        "expected_sources": ["reference/gcp-azure-metrics-comparison.md"],
        "expected_sections": [
            ("reference/gcp-azure-metrics-comparison.md", "OpenTelemetry support"),
            ("reference/gcp-azure-metrics-comparison.md", "Native agents"),
        ],
        "expected_facts": ["Ops Agent", "Azure Monitor Agent"],
    },
    {
        "id": "tagging-standard",
        "question": "What is the resource tagging standard?",
        "expected_sources": ["reference/tagging-standard.md"],
        "expected_sections": [("reference/tagging-standard.md", "Required tags")],
        "expected_facts": ["owner", "cost-centre", "environment"],
    },
    {
        "id": "tracing-use-cases",
        "question": "What are some use cases for distributed tracing?",
        "expected_sources": ["architecture/observability_tracing_use_cases.md"],
        "expected_sections": [
            ("architecture/observability_tracing_use_cases.md", "Latency investigation"),
            ("architecture/observability_tracing_use_cases.md", "Dependency mapping"),
        ],
        "expected_facts": ["dependency", "latency"],
    },
    {
        "id": "data-first-approach",
        "question": "How does the data-first approach avoid unnecessary cloud API calls by checking data freshness first?",
        "expected_sources": ["data-first-approach.md"],
        "expected_sections": [("data-first-approach.md", "Freshness check")],
        "expected_facts": ["6 hours"],
    },
    {
        "id": "observability-setup",
        "question": "According to the observability setup primer, what are the prerequisites and steps for onboarding a new service?",
        "expected_sources": ["observability-setup-primer.md"],
        "expected_sections": [
            ("observability-setup-primer.md", "Prerequisites"),
            ("observability-setup-primer.md", "Onboarding steps"),
        ],
        "expected_facts": ["owning team", "SLO"],
    },
    {
        "id": "cost-smoothing",
        "question": "What is the cost smoothing plan for Log Analytics?",
        "expected_sources": ["programme/COST_SMOOTHING.md"],
        "expected_sections": [("programme/COST_SMOOTHING.md", "Plan")],
        "expected_facts": ["20%", "three waves"],
    },
    # Exact-term questions: figures, product names and tag names that
    # MiniLM embeddings match less reliably than keywords (issue #8).
    {
        "id": "exact-gcp-coverage-figure",
        "question": "Which cloud is at 0.47% coverage?",
        "expected_sources": ["reference/coverage-targets.md"],
        "expected_sections": [("reference/coverage-targets.md", "Current status")],
        "expected_facts": ["GCP"],
    },
    {
        "id": "exact-ops-agent",
        "question": "What does the Ops Agent collect and how is it configured?",
        "expected_sources": ["reference/gcp-azure-metrics-comparison.md"],
        "expected_sections": [("reference/gcp-azure-metrics-comparison.md", "Native agents")],
        "expected_facts": ["host metrics", "YAML"],
    },
    {
        "id": "exact-tag-values",
        "question": "What values are allowed for the environment and tier tags?",
        "expected_sources": ["reference/tagging-standard.md"],
        "expected_sections": [("reference/tagging-standard.md", "Allowed values")],
        "expected_facts": ["prod", "staging", "dev"],
    },
    {
        "id": "exact-tag-enforcement",
        "question": "What happens to resources missing the owner or cost-centre tag?",
        "expected_sources": ["reference/tagging-standard.md"],
        "expected_sections": [("reference/tagging-standard.md", "Enforcement")],
        "expected_facts": ["Azure Policy", "30 days"],
    },
    {
        "id": "exact-basic-logs",
        "question": "Which Log Analytics tables move to the Basic Logs tier?",
        "expected_sources": ["programme/COST_SMOOTHING.md"],
        "expected_sections": [("programme/COST_SMOOTHING.md", "Plan")],
        "expected_facts": ["high-volume", "30 days"],
    },
]

# Questions on observability-adjacent topics that the sample corpus doesn't
# cover. The right answer is to decline, not to invent a rota or a price.
UNANSWERABLE_EVALS = [
    {
        "id": "unanswerable-oncall-rota",
        "question": "Who is on the on-call rota for the payments team this week?",
    },
    {
        "id": "unanswerable-vendor-pricing",
        "question": "How much does a Datadog Pro licence cost per host per month?",
    },
    {
        "id": "unanswerable-k8s-version",
        "question": "Which Kubernetes version does the production AKS cluster run?",
    },
    {
        "id": "unanswerable-outage-root-cause",
        "question": "What was the root cause of last month's checkout outage?",
    },
]


def contains_fact(answer: str, fact: str) -> bool:
    """Case-insensitive match that must start at a word boundary, so that
    "4 months" doesn't match "14 months" and "5%" doesn't match "85%".
    The end is left open so "SLO" still matches "SLOs"."""
    return re.search(r"(?<!\w)" + re.escape(fact), answer, re.IGNORECASE) is not None


# Phrases that signal the answer declines rather than answering. Includes the
# engine's own replies when nothing is retrieved or the model refuses.
DECLINE_RE = re.compile(
    r"\b(?:does not|doesn't|do not|don't) (?:contain|include|mention|cover|provide|specify|say|have)"
    r"|\bnot (?:in|included in|contained in|covered (?:in|by)|mentioned in|available in|present in)"
    r" the (?:provided |given |retrieved )?(?:context|documents?|sources?|chunks?)"
    r"|\bno (?:information|mention|details?|data)\b"
    r"|\b(?:cannot|can't|unable to) (?:answer|determine|find|tell)"
    r"|\bno relevant documents found"
    r"|\bthe model declined",
    re.IGNORECASE,
)


def is_decline(answer: str) -> bool:
    """True if the answer says the context doesn't hold the answer."""
    return DECLINE_RE.search(answer) is not None


# A citation parenthesis, e.g. "(source: reference/coverage-targets.md, section '…')".
# Models sometimes list several documents in one: "(source: a.md; b.md)".
CITATION_RE = re.compile(r"\(\s*sources?:[^)]*\)", re.IGNORECASE)
DOCUMENT_RE = re.compile(r"[\w./-]+\.md\b")


def cited_sources(answer: str) -> set[str]:
    """Documents the answer cites with the (source: X.md, ...) format."""
    return {doc for citation in CITATION_RE.findall(answer)
            for doc in DOCUMENT_RE.findall(citation)}


def same_document(a: str, b: str) -> bool:
    """Whether two source paths name the same document, allowing either one
    to drop leading directories ("coverage-targets.md" vs
    "reference/coverage-targets.md")."""
    return a == b or a.endswith("/" + b) or b.endswith("/" + a)


def citation_problems(answer: str, retrieved_sources: set[str],
                      expected_sources: list[str],
                      citations: list[dict] | None = None) -> list[str]:
    """Return what's wrong with the answer's citations; empty if they're fine.

    Uses the structured citations when the provider returned them (Anthropic),
    otherwise parses the citations written in the answer text (DeepSeek).
    Fails when the answer cites nothing, cites a document that wasn't
    retrieved (so the citation can't be backed by the context), or never
    cites the document the answer should come from.
    """
    if citations is not None:
        cited = {c["source"] for c in citations}
    else:
        cited = cited_sources(answer)
    if not cited:
        return ["no citations"]
    problems = []
    unretrieved = sorted(c for c in cited
                         if not any(same_document(c, r) for r in retrieved_sources))
    if unretrieved:
        problems.append(f"cites sources that weren't retrieved: {unretrieved}")
    uncited = [e for e in expected_sources
               if not any(same_document(c, e) for c in cited)]
    if uncited:
        problems.append(f"doesn't cite expected sources: {uncited}")
    return problems


RETRIEVE_N = 6  # number of chunks to retrieve for document and section checks


def supporting_section_ranks(chunks: list[dict],
                             expected_sections: list[tuple[str, str]]) -> list[int | None]:
    """First 1-based rank for each required source/section, or None if absent."""
    return [
        next((rank for rank, chunk in enumerate(chunks, start=1)
              if (chunk["source"], chunk.get("section")) == (source, section)), None)
        for source, section in expected_sections
    ]


def run_retrieval_evals(engine: RAGEngine, verbose: bool) -> tuple[int, int]:
    """Run retrieval-only evals — no LLM API call needed."""
    passed = 0
    failed = 0
    source_passed = 0
    section_hits = dict.fromkeys((1, 3, RETRIEVE_N), 0)
    section_total = 0

    print(f"Retrieval evals (documents and supporting sections in top-{RETRIEVE_N})")
    print("=" * 60)

    for ev in EVALS:
        chunks = engine.retrieve(ev["question"], n_results=RETRIEVE_N)
        retrieved_sources = {c["source"] for c in chunks}

        # Check that all expected sources appear in the retrieved set
        expected = ev["expected_sources"]
        missing = [s for s in expected if s not in retrieved_sources]
        sources_ok = not missing
        source_passed += sources_ok
        ranks = supporting_section_ranks(chunks, ev["expected_sections"])
        section_total += len(ranks)
        for cutoff in section_hits:
            section_hits[cutoff] += sum(rank is not None and rank <= cutoff for rank in ranks)
        sections_ok = all(rank is not None and rank <= RETRIEVE_N for rank in ranks)
        first_rank = min((rank for rank in ranks if rank is not None), default=None)
        ok = sources_ok and sections_ok

        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {ev['id']}  sources={'ok' if sources_ok else 'FAIL'} "
              f"sections={'ok' if sections_ok else 'FAIL'} "
              f"first relevant rank={first_rank if first_rank is not None else 'missing'}")

        if not ok or verbose:
            print(f"       Q: {ev['question']}")
            print(f"       Retrieved: {', '.join(sorted(retrieved_sources))}")
            if missing:
                print(f"       Missing sources: {missing}")
            for (source, section), rank in zip(ev["expected_sections"], ranks, strict=True):
                print(f"       {source} / {section}: rank {rank if rank is not None else 'missing'}")

        if ok:
            passed += 1
        else:
            failed += 1

    print(f"\nDocument attribution: {source_passed}/{len(EVALS)} cases")
    print("Supporting-section recall: " + ", ".join(
        f"@{cutoff} {hits}/{section_total}" for cutoff, hits in section_hits.items()))

    # Unanswerable questions: nothing should survive the distance cutoff
    for ev in UNANSWERABLE_EVALS:
        chunks = engine.retrieve(ev["question"], n_results=RETRIEVE_N)
        ok = not chunks

        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {ev['id']}")

        if not ok or verbose:
            print(f"       Q: {ev['question']}")
            for c in chunks:
                print(f"       Kept: {c['source']} (dist {c['distance']:.3f})")

        if ok:
            passed += 1
        else:
            failed += 1

    return passed, failed


def run_full_evals(engine: RAGEngine, verbose: bool) -> tuple[int, int]:
    """Run full evals — retrieval + LLM answer + fact and citation checks."""
    passed = 0
    failed = 0
    dimension_failures = {"sources": 0, "facts": 0, "citations": 0, "declines": 0}
    answerable_errors = 0
    unanswerable_errors = 0

    print("\nFull evals (retrieval + LLM answer + fact presence + citations)")
    print("=" * 60)

    for ev in EVALS:
        try:
            answer, chunks, citations = engine.answer_with_citations(
                ev["question"], n_results=N_RESULTS, verbose=False,
            )
        except Exception as e:  # noqa: BLE001 — any provider/SDK error fails the case
            print(f"[FAIL] {ev['id']} — LLM error: {e}")
            failed += 1
            answerable_errors += 1
            continue

        # Check 1: source attribution in retrieved chunks
        retrieved_sources = {c["source"] for c in chunks}
        expected = ev["expected_sources"]
        found = [s for s in expected if any(s in r for r in retrieved_sources)]
        missing_sources = [s for s in expected if s not in found]
        sources_ok = len(missing_sources) == 0

        # Check 2: fact presence in the answer
        missing_facts = [f for f in ev["expected_facts"] if not contains_fact(answer, f)]
        facts_ok = not missing_facts

        # Check 3: the answer's citations (structured when the provider has them)
        bad_citations = citation_problems(answer, retrieved_sources, expected, citations)
        citations_ok = not bad_citations

        dimension_failures["sources"] += not sources_ok
        dimension_failures["facts"] += not facts_ok
        dimension_failures["citations"] += not citations_ok

        ok = sources_ok and facts_ok and citations_ok
        status = "PASS" if ok else "FAIL"
        dims = " ".join(f"{name}={'ok' if good else 'FAIL'}" for name, good in
                        [("sources", sources_ok), ("facts", facts_ok),
                         ("citations", citations_ok)])
        print(f"[{status}] {ev['id']}  {dims}")

        if not ok or verbose:
            print(f"       Q: {ev['question']}")
            print(f"       Retrieved: {', '.join(sorted(retrieved_sources))}")
            if missing_sources:
                print(f"       Missing sources: {missing_sources}")
            if missing_facts:
                print(f"       Missing facts: {missing_facts}")
            for problem in bad_citations:
                print(f"       Citations: {problem}")
            print(f"       A: {answer[:300]}")

        if ok:
            passed += 1
        else:
            failed += 1

    # Unanswerable questions run without the distance cutoff, so the model
    # sees the nearest chunks and has to decline on its own.
    for ev in UNANSWERABLE_EVALS:
        try:
            answer, chunks = engine.answer(
                ev["question"], n_results=N_RESULTS, verbose=False,
                max_distance=None,
            )
        except Exception as e:  # noqa: BLE001 — any provider/SDK error fails the case
            print(f"[FAIL] {ev['id']} — LLM error: {e}")
            failed += 1
            unanswerable_errors += 1
            continue

        ok = is_decline(answer)
        dimension_failures["declines"] += not ok
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {ev['id']}  declines={'ok' if ok else 'FAIL'}")

        if not ok or verbose:
            print(f"       Q: {ev['question']}")
            print(f"       Retrieved: {', '.join(sorted({c['source'] for c in chunks}))}")
            print(f"       A: {answer[:300]}")

        if ok:
            passed += 1
        else:
            failed += 1

    totals = {"sources": len(EVALS), "facts": len(EVALS), "citations": len(EVALS),
              "declines": len(UNANSWERABLE_EVALS)}
    unevaluated = {"sources": answerable_errors, "facts": answerable_errors,
                   "citations": answerable_errors, "declines": unanswerable_errors}
    print(f"\nProvider errors: {answerable_errors + unanswerable_errors}/{passed + failed}")
    print("\nBy dimension: " + ", ".join(
        f"{name} {totals[name] - unevaluated[name] - dimension_failures[name]}/"
        f"{totals[name] - unevaluated[name]} ({unevaluated[name]} unevaluated)"
        for name in totals))

    return passed, failed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG evals")
    parser.add_argument("--with-llm", action="store_true",
                        help="Also run LLM-based fact-presence, citation and decline checks")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--persist-dir", default="chroma_db",
                        help="Chroma persistence directory")
    parser.add_argument("--retrieval", default=DEFAULT_RETRIEVAL, choices=RETRIEVAL_MODES,
                        help=f"Retrieval mode to evaluate (default: {DEFAULT_RETRIEVAL})")
    args = parser.parse_args()

    engine = RAGEngine(persist_dir=args.persist_dir, retrieval=args.retrieval)
    print(f"Retrieval mode: {args.retrieval}\n")

    p1, f1 = run_retrieval_evals(engine, verbose=args.verbose)
    print(f"\nRetrieval results: {p1}/{p1 + f1} passed")

    if args.with_llm:
        p2, f2 = run_full_evals(engine, verbose=args.verbose)
        print(f"Full results:       {p2}/{p2 + f2} passed")
        print(f"\nCombined: {p1 + p2}/{p1 + p2 + f1 + f2} passed")
        failures = f1 + f2
    else:
        print("(Run with --with-llm to also check facts, citations and declines in generated answers)")
        failures = f1

    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
