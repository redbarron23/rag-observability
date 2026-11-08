"""The #34 keyword baseline and benchmark helpers, without an index or model."""

import pytest

import benchmark
import evals
from benchmark import KeywordRetriever


def chunk(chunk_id, text, source="doc.md", section="S"):
    return {"id": chunk_id, "text": text, "source": source, "file": source, "section": section}


@pytest.fixture
def retriever():
    return KeywordRetriever([
        chunk("a#0", "Tier 1 production resources target 85% coverage within 4 months.",
              section="Tier 1"),
        chunk("b#0", "Basic Logs tables keep data for 30 days.", source="cost.md",
              section="Basic Logs"),
        chunk("c#0", "The Ops Agent collects metrics and logs on GCP VMs.", source="gcp.md",
              section="Ops Agent"),
    ])


def test_ranks_by_bm25_and_keeps_chunk_metadata(retriever):
    hits = retriever.retrieve("Which tables move to Basic Logs?", n_results=6)
    assert hits[0]["id"] == "b#0"
    assert hits[0]["source"] == "cost.md" and hits[0]["section"] == "Basic Logs"
    assert hits[0]["score"] > 0
    assert hits[0]["distance"] is None  # keyword has no cosine distance or cutoff


def test_exact_figures_and_percentages_match_as_tokens(retriever):
    assert retriever.retrieve("What does 85% apply to?")[0]["id"] == "a#0"


def test_chunks_without_a_shared_term_are_never_returned(retriever):
    assert retriever.retrieve("kubernetes version") == []


def test_n_results_limits_the_ranking(retriever):
    assert len(retriever.retrieve("logs metrics coverage", n_results=1)) == 1


def test_scores_with_the_existing_retrieval_evals(monkeypatch, retriever):
    monkeypatch.setattr(evals, "EVALS", [{
        "id": "basic-logs", "question": "How long do Basic Logs tables keep data?",
        "expected_sources": ["cost.md"], "expected_sections": [("cost.md", "Basic Logs")],
        "expected_facts": ["30 days"],
    }])
    # No cutoff, so an off-topic question sharing a word still retrieves: the
    # baseline can't decline, and the unanswerable check must say so.
    monkeypatch.setattr(evals, "UNANSWERABLE_EVALS", [{
        "id": "off-topic", "question": "Which logs does the rota mention?"}])
    records = []
    assert evals.run_retrieval_evals(retriever, verbose=True, records=records) == (1, 1)
    summary = benchmark.summarise(records, evals.RETRIEVE_N)
    assert summary["answerable"]["passed"] == 1
    assert summary["answerable"]["supporting_section_recall"] == {
        "total": 1, "at_1": 1, "at_3": 1, "at_6": 1}
    assert summary["answerable"]["mrr_first_supporting_section"] == 1.0
    assert summary["unanswerable"] == {"rejected": 0, "total": 1, "failed_cases": ["off-topic"]}


def test_latency_summary_reports_median_and_p95():
    stats = benchmark.latency_summary([float(ms) for ms in range(1, 101)])
    assert stats["queries"] == 100
    assert stats["median_ms"] == 50.5
    assert stats["p95_ms"] == 95.0
    assert stats["max_ms"] == 100.0
