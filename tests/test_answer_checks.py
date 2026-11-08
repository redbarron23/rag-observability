"""Regression fixtures for the stricter answer checks (issue #29). Offline."""

from pathlib import Path
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


# Issue #104. The sentence that failed unanswerable-logging-spend-amount in
# the 8 November 2026 vector run (docs/benchmarks/generation-vector-2026-11-08.json),
# verbatim from the recorded problem, including the citation the sentence
# splitter carried over from the sentence before it. The run did not record
# the full answer (no --include-answers), so the decline before it is a
# reconstruction.
SPEND = next(e for e in evals.UNANSWERABLE_EVALS
             if e["id"] == "unanswerable-logging-spend-amount")
SPEND_CORPUS = (Path(__file__).resolve().parent.parent
                / "data" / "runbooks" / "logging-cost-spike.md").read_text()
ISSUE_104_SENTENCE = (
    "(source: runbooks/logging-cost-spike.md, section 'Investigation: GCP Cloud Logging') "
    "Running that query for project-bravo on 23 April should give the cost."
)
ISSUE_104_LEAD = (
    "The documents don't contain logging spend figures for project-bravo on 23 April. "
    "The runbook says to take amounts from the billing data and describes filtering the "
    "billing export by Cloud Logging and grouping cost by project and day."
)


def test_issue_104_next_step_with_question_details_passes():
    answer = f"{ISSUE_104_LEAD} {ISSUE_104_SENTENCE}"
    assert decline_problems(answer, SPEND["question"], SPEND_CORPUS) == []


def test_issue_104_failed_on_the_opening_word_not_the_question_details():
    """The old check flagged "Running", capitalised only because it opens the
    sentence. project-bravo and 23 April were already known from the
    question. The fix is structural: an opening word followed by a
    determiner heads a verb phrase, so its capital says nothing."""
    known = {w.strip(".-").lower()
             for w in evals.WORD_RE.findall(f"{SPEND['question']} {SPEND_CORPUS}")}
    assert {"project-bravo", "23", "april"} <= known
    assert "running" not in known and "running" not in evals.SENTENCE_STARTERS
    assert ISSUE_104_SENTENCE[evals.verb_opening(ISSUE_104_SENTENCE):].startswith("Running")


@pytest.mark.parametrize(("sentence", "invented"), [
    # An invented amount after the decline.
    ("Project-bravo spent $412.50 on logging on 23 April.", "412.50"),
    ("Running that query for project-bravo on 23 April should show about $40.", "40"),
    # An invented date: the same next step, on a day the question didn't ask about.
    ("Running that query for project-bravo on 24 April should give the cost.", "24"),
    # An invented figure in a verb-led sentence.
    ("Expect the cost to be roughly 3.2 times the usual daily spend.", "3.2"),
    # A name that opens a sentence is still a name.
    ("Priya is the FinOps owner for project-bravo.", "Priya"),
])
def test_issue_104_invented_specifics_after_a_decline_still_fail(sentence, invented):
    answer = f"{ISSUE_104_LEAD} {sentence}"
    problems = decline_problems(answer, SPEND["question"], SPEND_CORPUS)
    assert len(problems) == 1 and invented in problems[0]


def test_lowercase_invented_identifier_is_not_caught():
    """Documented limitation (docs/guide.md, Dimension 3): a specific counts
    only if it has a digit or a capital, so a lowercase identifier that is not
    in the question ("project-delta") passes. This predates #104."""
    answer = (f"{ISSUE_104_LEAD} Running that query for project-delta on 23 April "
              "should give the cost.")
    assert decline_problems(answer, SPEND["question"], SPEND_CORPUS) == []


@pytest.mark.parametrize(("sentence", "opens_verb_phrase"), [
    ("Run the billing export query.", True),
    ("Filter your export by project.", True),
    ('"Using the runbook query, group by day.', True),
    ("Alice is on call this week.", False),
    ("GCP the platform is covered.", False),  # acronyms never count as verbs
    ("Datadog Pro costs $15 per host.", False),
])
def test_verb_opening(sentence, opens_verb_phrase):
    assert (evals.verb_opening(sentence) is not None) == opens_verb_phrase


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
