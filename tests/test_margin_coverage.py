"""Exact-term margin coverage reporting and floor. Offline, fixed chunks."""

from unittest.mock import Mock

import pytest

import eval_artifact
import evals
from query import kept_by_exact_term

CASE = {"id": "named-thing", "question": "What does the Ops Agent do?",
        "expected_sources": ["doc.md"], "expected_sections": [("doc.md", "Agents")],
        "expected_facts": ["YAML"]}


def chunk(distance, section="Agents", **extra):
    return {"id": f"doc.md#{section}", "source": "doc.md", "section": section,
            "text": "The Ops Agent uses YAML.", "distance": distance, **extra}


def test_kept_by_exact_term():
    assert kept_by_exact_term(chunk(0.64), 0.6)
    assert not kept_by_exact_term(chunk(0.55), 0.6)
    assert not kept_by_exact_term(chunk(0.64, expanded_from="doc.md#0"), 0.6)
    assert not kept_by_exact_term(chunk(0.64), None)
    assert not kept_by_exact_term(chunk(None), 0.6)


def test_margin_support_only_counts_supporting_chunks():
    chunks = [chunk(0.64), chunk(0.65, section="Other")]
    assert evals.margin_support(chunks, CASE["expected_sections"]) == ["doc.md#Agents"]
    assert evals.margin_support([chunk(0.5)], CASE["expected_sections"]) == []


def run_main(monkeypatch, capsys, distance):
    monkeypatch.setattr("sys.argv", ["evals.py"])
    monkeypatch.setattr(evals, "EVALS", [CASE])
    monkeypatch.setattr(evals, "UNANSWERABLE_EVALS", [])
    monkeypatch.setattr(evals, "HELDOUT_EVALS", [])
    monkeypatch.setattr(evals, "HELDOUT_UNANSWERABLE_EVALS", [])
    engine = Mock()
    engine.retrieve.return_value = [chunk(distance)]
    monkeypatch.setattr(evals, "RAGEngine", lambda **kwargs: engine)
    with pytest.raises(SystemExit) as exc:
        evals.main()
    return exc.value.code, capsys.readouterr().out


def test_case_kept_by_margin_is_reported_and_passes(monkeypatch, capsys):
    code, out = run_main(monkeypatch, capsys, 0.64)
    assert code == 0
    assert "Exact-term margin kept supporting evidence for: named-thing" in out


def test_floor_fails_when_no_case_uses_the_margin(monkeypatch, capsys):
    # The case itself still passes (it's answerable); only the coverage floor fails.
    code, out = run_main(monkeypatch, capsys, 0.4)
    assert "[PASS] named-thing" in out
    assert "Exact-term margin kept supporting evidence for: no cases" in out
    assert "[FAIL] exact-term margin coverage" in out
    assert code == 1


def test_artifact_summary_lists_margin_cases():
    cases = [{"id": "a", "status": "pass", "dimensions": {}, "exact_term_margin": ["d#1"]},
             {"id": "b", "status": "pass", "dimensions": {}, "exact_term_margin": []}]
    assert eval_artifact.section(cases, 6)["summary"]["exact_term_margin_cases"] == ["a"]
    assert "exact_term_margin_cases" not in eval_artifact.section(
        [{"id": "c", "status": "pass", "dimensions": {}}], 6)["summary"]
