"""Retrieval scoring checks with fixed ranked chunks and no model or API calls."""

from unittest.mock import Mock

import pytest

import evals
from ingest import chunk_text_simple, load_documents


@pytest.mark.parametrize("ranked_sections, totals, recall, first_rank", [
    ([("doc.md", "Introduction")], (0, 1), (0, 0, 0), "missing"),
    ([("doc.md", "First"), ("other.md", "Second")], (0, 1), (1, 1, 1), "1"),
    ([("doc.md", "First"), ("doc.md", "First"), ("doc.md", "Second")],
     (1, 0), (1, 2, 2), "1"),
    ([("doc.md", "Introduction"), ("doc.md", "First"),
      ("doc.md", "Other"), ("doc.md", "Other"),
      ("doc.md", "Other"), ("doc.md", "Second")], (1, 0), (0, 1, 2), "2"),
    ([], (0, 1), (0, 0, 0), "missing"),
])
def test_retrieval_scores_required_sections_and_ranks(
    monkeypatch, capsys, ranked_sections, totals, recall, first_rank,
):
    monkeypatch.setattr(evals, "EVALS", [{
        "id": "two-sections", "question": "Explain both sections",
        "expected_sources": ["doc.md"],
        "expected_sections": [("doc.md", "First"), ("doc.md", "Second")],
    }])
    monkeypatch.setattr(evals, "UNANSWERABLE_EVALS", [])
    engine = Mock()
    engine.retrieve.return_value = [
        {"source": source, "section": section} for source, section in ranked_sections
    ]

    assert evals.run_retrieval_evals(engine, verbose=True) == totals
    engine.retrieve.assert_called_once_with("Explain both sections", n_results=6)
    output = capsys.readouterr().out
    source_passes = int(any(source == "doc.md" for source, _ in ranked_sections))
    assert f"Document attribution: {source_passes}/1 cases" in output
    assert f"first relevant rank={first_rank}" in output
    for cutoff, hits in zip((1, 3, 6), recall, strict=True):
        assert f"@{cutoff} {hits}/2" in output
    if not totals[0]:
        assert "doc.md / Second: rank missing" in output


def test_section_labels_are_scoped_to_source_and_use_first_matching_chunk():
    chunks = [
        {"source": "other.md", "section": "Target"},
        {"source": "doc.md"},
        {"source": "doc.md", "section": "Target"},
        {"source": "doc.md", "section": "Target"},
    ]
    assert evals.supporting_section_ranks(
        chunks, [("doc.md", "Target"), ("doc.md", "Absent")],
    ) == [3, None]


def test_corpus_section_annotations_exist_and_cover_expected_sources():
    sections = {
        (doc["rel_path"], chunk["section"])
        for doc in load_documents("data")
        for chunk in chunk_text_simple(doc["text"], chunk_size=500)
    }
    for case in evals.EVALS:
        assert case["expected_sections"], case["id"]
        assert set(case["expected_sections"]) <= sections, case["id"]
        assert {source for source, _ in case["expected_sections"]} == set(case["expected_sources"])


def test_missing_supporting_section_fails_cli(monkeypatch):
    monkeypatch.setattr("sys.argv", ["evals.py"])
    monkeypatch.setattr(evals, "EVALS", [{
        "id": "missing-section", "question": "Target?",
        "expected_sources": ["doc.md"], "expected_sections": [("doc.md", "Target")],
    }])
    monkeypatch.setattr(evals, "UNANSWERABLE_EVALS", [])
    engine = Mock()
    engine.retrieve.return_value = [{"source": "doc.md", "section": "Introduction"}]
    monkeypatch.setattr(evals, "RAGEngine", lambda **kwargs: engine)
    with pytest.raises(SystemExit) as exc:
        evals.main()
    assert exc.value.code == 1
