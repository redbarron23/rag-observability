"""Regression fixtures for the stricter answer checks (issue #29). Offline."""

from unittest.mock import Mock

import pytest

import evals
from evals import (
    cited_sections,
    decline_problems,
    is_decline,
    normalise_section,
    relation_problems,
    section_problems,
)

TIER1 = next(e for e in evals.EVALS if e["id"] == "coverage-targets-tier1")
TIER1_CHUNKS = [{"source": "reference/coverage-targets.md", "section": "Tier 1",
                 "text": "The production Tier 1 coverage target is 85% within 4 months "
                         "and 100% within 9 months of programme start."}]

# The false positive reported in #29: every expected fact is present and the
# right document is cited, but the deadlines are swapped and the section is invented.
SWAPPED = ("Tier 1 is 100% within 4 months and 85% within 9 months "
           "(source: reference/coverage-targets.md, section 'Invented').")
# The answer Claude Sonnet 5.5 gave in the 2026-11-01 eval run.
CORRECT = ("The production Tier 1 coverage target is 85% within 4 months and 100% within "
           "9 months of programme start. (source: reference/coverage-targets.md, section 'Tier 1')")


def test_swapped_answer_still_passes_the_old_checks():
    # Documents why the stricter checks are needed.
    assert all(evals.contains_fact(SWAPPED, f) for f in TIER1["expected_facts"])
    assert evals.citation_problems(SWAPPED, {"reference/coverage-targets.md"},
                                   TIER1["expected_sources"]) == []


def test_swapped_deadlines_fail_the_relation_check():
    assert relation_problems(SWAPPED, TIER1["expected_relations"]) == [
        "'85%' not stated with '4 months'",
        "'100%' not stated with '9 months'",
    ]


def test_correct_answer_passes_the_relation_check():
    assert relation_problems(CORRECT, TIER1["expected_relations"]) == []
    split = "The target is 85% within 4 months. It rises to 100% within 9 months."
    assert relation_problems(split, TIER1["expected_relations"]) == []


def test_relation_check_is_order_sensitive_and_needs_one_sentence():
    reversed_order = "Within 4 months the target is 85%, and within 9 months 100%."
    assert relation_problems(reversed_order, TIER1["expected_relations"])  # documented false failure
    across_sentences = "The targets are 85% and 100%. The deadlines are 4 months and 9 months."
    assert len(relation_problems(across_sentences, TIER1["expected_relations"])) == 2


def test_invented_section_fails_the_section_check():
    assert section_problems(SWAPPED, TIER1_CHUNKS) == [
        "cites a section that wasn't retrieved: reference/coverage-targets.md / 'Invented'",
    ]
    assert section_problems(CORRECT, TIER1_CHUNKS) == []


@pytest.mark.parametrize("cited", [
    "Tier 1", "tier 1", "  Tier  1 ", "Coverage Targets > Tier 1", "Coverage Targets \u203a Tier 1",
])
def test_section_normalisation(cited):
    assert normalise_section(cited) == normalise_section("Tier 1") == "tier 1"


def test_section_check_needs_the_same_document_and_reads_combined_citations():
    answer = ("x (source: reference/coverage-targets.md, section 'Tier 1'; "
              "source: tagging-standard.md, section \u2018Owner\u2019)")
    assert cited_sections(answer) == [("reference/coverage-targets.md", "Tier 1"),
                                      ("tagging-standard.md", "Owner")]
    chunks = [*TIER1_CHUNKS, {"source": "reference/tagging-standard.md", "section": "Owner"}]
    assert section_problems(answer, chunks) == []
    # Right section name, wrong document.
    wrong = "x (source: reference/tagging-standard.md, section 'Tier 1')"
    assert section_problems(wrong, chunks)


def test_structured_citation_sections_are_checked():
    citations = [{"source": "reference/coverage-targets.md", "section": "Tier 9",
                  "chunk": 1, "cited_text": "..."}]
    assert section_problems("85%.", TIER1_CHUNKS, citations) == [
        "cites a section that wasn't retrieved: reference/coverage-targets.md / 'Tier 9'",
    ]


ROTA_Q = "Who is on the on-call rota for the payments team this week?"
GUESS = "The context does not contain a rota. Alice is on call this week."


def test_decline_followed_by_a_guess_is_caught():
    assert is_decline(GUESS)  # the old check accepted it
    problems = decline_problems(GUESS, ROTA_Q, corpus_text="")
    assert len(problems) == 1 and "Alice" in problems[0]


@pytest.mark.parametrize("answer", [
    # Real Claude Sonnet 5.5 declines from the 2026-11-01 run: they describe
    # what the documents do cover, using terms from the question or corpus.
    "The documents don't say who is on the on-call rota for the payments team this week. "
    "They contain no on-call schedules, rota information, or details about a payments team.",
    "The documents don't contain this information. None of them mention Datadog or its Pro "
    "licence pricing. The only cost-related content covers Log Analytics cost smoothing, "
    "commitment tiers, and tracing-based cost attribution.",
    "The documents don't say which Kubernetes version the production AKS cluster runs. "
    "They cover GCP vs. Azure metrics (native agents, retention and query) and coverage targets.",
    "No relevant documents found.",
    "The documents don't contain a rota. Check your PagerDuty schedule instead.",
])
def test_clean_declines_pass(answer):
    corpus = ("Log Analytics cost smoothing. Commitment tiers. GCP and Azure metrics. "
              "Coverage targets. OpenTelemetry.")
    question = ("Who is on the on-call rota for the payments team this week? "
                "How much does a Datadog Pro licence cost? "
                "Which Kubernetes version does the production AKS cluster run?")
    assert decline_problems(answer, question, corpus) == []


def test_non_decline_and_invented_numbers_fail():
    assert decline_problems("Datadog Pro costs $15 per host.", "Datadog Pro price?", "") == [
        "does not decline"]
    problems = decline_problems(
        "The documents don't say. The cluster runs version 1.29.",
        "Which Kubernetes version does the cluster run?", "")
    assert problems and "1.29" in problems[0]


def test_full_evals_apply_the_new_checks(monkeypatch, capsys):
    monkeypatch.setattr(evals, "EVALS", [TIER1])
    monkeypatch.setattr(evals, "UNANSWERABLE_EVALS", [{"id": "rota", "question": ROTA_Q}])

    class Engine:
        def __init__(self):
            self.collection = Mock()
            self.collection.get.return_value = {"documents": ["Coverage targets per tier."]}

        def answer_with_citations(self, question, **kwargs):
            return SWAPPED, TIER1_CHUNKS, None

        def answer(self, question, **kwargs):
            return GUESS, []

    assert evals.run_full_evals(Engine(), verbose=False) == (0, 2)
    out = capsys.readouterr().out
    assert "facts=FAIL citations=FAIL" in out
    assert "'Invented'" in out and "Alice" in out


def test_heldout_results_do_not_change_the_exit_code(monkeypatch, capsys):
    monkeypatch.setattr(evals, "REQUIRE_MARGIN_COVERAGE", False)
    monkeypatch.setattr("sys.argv", ["evals.py"])
    monkeypatch.setattr(evals, "RAGEngine", lambda **kwargs: object())
    calls = []

    def fake(engine, verbose, answerable=None, unanswerable=None, records=None):
        calls.append(answerable)
        return (0, 3) if answerable is evals.HELDOUT_EVALS else (5, 0)

    monkeypatch.setattr(evals, "run_retrieval_evals", fake)
    with pytest.raises(SystemExit) as exc:
        evals.main()
    assert exc.value.code == 0
    assert calls == [None, evals.HELDOUT_EVALS]
    assert "Held-out retrieval: 0/3 passed" in capsys.readouterr().out
