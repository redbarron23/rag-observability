"""Unit tests for provider validation and eval answer checks (no API calls)."""

import pytest

from evals import citation_problems, cited_sources, contains_fact, is_decline
from query import RAGEngine


def test_fact_must_start_at_word_boundary():
    assert contains_fact("Target is 85% within 4 months.", "4 months")
    assert not contains_fact("Target is 85% within 14 months.", "4 months")
    assert not contains_fact("Target is 85%.", "5%")


def test_fact_match_is_case_insensitive_and_allows_suffixes():
    assert contains_fact("Owner and Cost-Centre tags", "cost-centre")
    assert contains_fact("Each service defines SLOs", "SLO")


def test_cited_sources_parses_single_and_combined_citations():
    answer = (
        "Tier 1 is 85% (source: reference/coverage-targets.md, section 'Tier 1'). "
        "Tags are required (source: tagging-standard.md, section 'Tags'; "
        "source: `architecture/observability_framework.md`, section 'Ownership'). "
        "Costs are smoothed (source: programme/COST_SMOOTHING.md; data-first-approach.md)."
    )
    assert cited_sources(answer) == {
        "reference/coverage-targets.md",
        "tagging-standard.md",
        "architecture/observability_framework.md",
        "programme/COST_SMOOTHING.md",
        "data-first-approach.md",
    }


def test_citations_pass_when_expected_source_is_cited_and_retrieved():
    retrieved = {"reference/coverage-targets.md", "reference/tagging-standard.md"}
    # The model may drop the directory from the source path
    answer = "Tier 1 is 85% (source: coverage-targets.md, section 'Tier 1')."
    assert citation_problems(answer, retrieved, ["reference/coverage-targets.md"]) == []


def test_missing_citation_fails():
    assert citation_problems("Tier 1 is 85%.", {"reference/coverage-targets.md"},
                             ["reference/coverage-targets.md"]) == ["no citations"]


def test_citing_unretrieved_or_wrong_source_fails():
    retrieved = {"reference/coverage-targets.md", "reference/tagging-standard.md"}
    expected = ["reference/coverage-targets.md"]

    unretrieved = "Tier 1 is 85% (source: data-first-approach.md, section 'Freshness')."
    problems = citation_problems(unretrieved, retrieved, expected)
    assert any("weren't retrieved" in p for p in problems)

    wrong_doc = "Tier 1 is 85% (source: reference/tagging-standard.md, section 'Tags')."
    assert citation_problems(wrong_doc, retrieved, expected) == [
        "doesn't cite expected sources: ['reference/coverage-targets.md']"
    ]


@pytest.mark.parametrize("answer", [
    "The provided context does not contain information about the on-call rota.",
    "This isn't covered: the answer is not in the provided context.",
    "I cannot answer this from the documents given.",
    "No relevant documents found.",
    "The model declined to answer this question.",
])
def test_declines_are_recognised(answer):
    assert is_decline(answer)


def test_an_invented_answer_is_not_a_decline():
    assert not is_decline("Datadog Pro costs $15 per host per month.")


def test_unknown_provider_is_rejected(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "antropic")
    engine = object.__new__(RAGEngine)  # skip loading Chroma and the embedder
    with pytest.raises(ValueError, match="Unknown LLM provider 'antropic'"):
        engine._call_llm("question", "context")
