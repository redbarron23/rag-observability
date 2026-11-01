"""Unit tests for provider validation and eval answer checks (no API calls)."""

import pytest

import evals
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
        engine._call_llm("question", [])


@pytest.mark.parametrize("error_questions, source_score, decline_score, totals", [
    ({"answerable-pass", "answerable-fail", "decline-pass", "decline-fail"},
     "0/0 (2 unevaluated)", "0/0 (2 unevaluated)", (0, 4)),
    ({"answerable-fail", "decline-fail"},
     "1/1 (1 unevaluated)", "1/1 (1 unevaluated)", (2, 2)),
    ({"answerable-pass"}, "0/1 (1 unevaluated)", "1/2 (0 unevaluated)", (1, 3)),
    ({"decline-pass"}, "1/2 (0 unevaluated)", "0/1 (1 unevaluated)", (1, 3)),
    (set(), "1/2 (0 unevaluated)", "1/2 (0 unevaluated)", (2, 2)),
])
def test_full_eval_summary_separates_errors_from_evaluated_checks(
    monkeypatch, capsys, error_questions, source_score, decline_score, totals,
):
    monkeypatch.setattr(evals, "EVALS", [
        {"id": q, "question": q, "expected_sources": ["doc.md"],
         "expected_facts": ["85%"]}
        for q in ("answerable-pass", "answerable-fail")
    ])
    monkeypatch.setattr(evals, "UNANSWERABLE_EVALS", [
        {"id": q, "question": q} for q in ("decline-pass", "decline-fail")
    ])

    class Engine:
        def answer(self, question, **kwargs):
            if question in error_questions:
                raise RuntimeError("provider unavailable")
            if question == "answerable-pass":
                return ("85% (source: doc.md, section 'Target').",
                        [{"source": "doc.md", "section": "Target"}])
            if question == "decline-pass":
                return "No relevant documents found.", []
            return "An unsupported guess.", []

        def answer_with_citations(self, question, **kwargs):
            return (*self.answer(question, **kwargs), None)  # prompt-written citations

    assert evals.run_full_evals(Engine(), verbose=False) == totals
    output = capsys.readouterr().out
    assert f"Provider errors: {len(error_questions)}/4" in output
    for name in ("sources", "facts", "citations"):
        assert f"{name} {source_score}" in output
    assert f"declines {decline_score}" in output


def test_full_eval_errors_preserve_nonzero_cli_exit(monkeypatch):
    monkeypatch.setattr("sys.argv", ["evals.py", "--with-llm"])
    monkeypatch.setattr(evals, "RAGEngine", lambda **kwargs: object())
    monkeypatch.setattr(evals, "run_retrieval_evals", lambda *args, **kwargs: (14, 0))
    monkeypatch.setattr(evals, "run_full_evals", lambda *args, **kwargs: (0, 14))
    with pytest.raises(SystemExit) as exc:
        evals.main()
    assert exc.value.code == 1
