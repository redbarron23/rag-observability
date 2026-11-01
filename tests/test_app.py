"""Exercise the Streamlit interface without loading models or calling providers."""

from pathlib import Path
from unittest.mock import Mock

import pytest
import streamlit as st
from chromadb.errors import NotFoundError
from streamlit.testing.v1 import AppTest

CHUNK = {
    "source": "reference/coverage-targets.md", "section": "Tier 1",
    "distance": 0.25, "text": "The Tier 1 target is 85% within 4 months.",
}
CITATION = {
    "source": "reference/coverage-targets.md", "section": "Tier 1", "chunk": 1,
    "cited_text": "The Tier 1 target is\n85% within 4 months.",
}
ANSWER = "85% within 4 months (source: coverage-targets.md)."


@pytest.fixture
def ui(monkeypatch):
    st.cache_resource.clear()
    engine = Mock()
    engine.retrieve.return_value = [CHUNK]
    engine.answer_with_citations.return_value = (ANSWER, [CHUNK], [CITATION])
    factory = Mock(return_value=engine)
    monkeypatch.setattr("query.RAGEngine", factory)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()
    yield app, engine, factory
    st.cache_resource.clear()


def submit(app, label):
    next(button for button in app.button if button.label == label).click().run()
    assert not app.exception


def test_retrieval_works_without_key_and_retains_evidence_on_rerun(ui):
    app, engine, factory = ui
    factory.assert_not_called()
    submit(app, "Find sources")
    factory.assert_called_once_with(persist_dir="chroma_db", retrieval="vector")
    engine.retrieve.assert_called_once_with(
        "What is the coverage target for Tier 1 production resources?",
        n_results=6, max_distance=0.6,
    )
    engine.answer_with_citations.assert_not_called()
    assert "Tier 1" in app.expander[0].label
    assert "distance 0.250" in app.expander[0].label
    assert app.text[0].value == CHUNK["text"]
    app.sidebar.slider[0].set_value(3).run()
    engine.retrieve.assert_called_once()
    assert app.text[0].value == CHUNK["text"]
    assert any("vector retrieval · top 6" in caption.value for caption in app.caption)


def test_generation_shows_quoted_citations(ui):
    app, engine, _ = ui
    submit(app, "Generate answer")
    engine.answer_with_citations.assert_called_once_with(
        "What is the coverage target for Tier 1 production resources?",
        n_results=6, max_distance=0.6,
    )
    engine.retrieve.assert_not_called()
    assert any(element.value == ANSWER for element in app.markdown)
    citations = app.expander[0]
    assert citations.label == "Quoted citations (1)"
    assert "[1] reference/coverage-targets.md, section 'Tier 1'" in citations.markdown[0].value
    assert citations.text[0].value == "The Tier 1 target is 85% within 4 months."
    assert "Tier 1" in app.expander[1].label


@pytest.mark.parametrize("citations", [None, []])
def test_generation_without_structured_citations_shows_no_quotes(ui, citations):
    app, engine, _ = ui
    engine.answer_with_citations.return_value = (ANSWER, [CHUNK], citations)
    submit(app, "Generate answer")
    assert any(element.value == ANSWER for element in app.markdown)
    assert not any("Quoted citations" in expander.label for expander in app.expander)
    assert len(app.expander) == 1


def test_retrieval_mode_selector_defaults_and_caches_engine_per_mode(ui):
    app, engine, factory = ui
    selector = app.sidebar.selectbox[0]
    assert selector.label == "Retrieval mode"
    assert selector.value == "vector"
    assert list(selector.options) == ["vector", "hybrid"]
    selector.set_value("hybrid").run()
    submit(app, "Find sources")
    factory.assert_called_once_with(persist_dir="chroma_db", retrieval="hybrid")
    assert any("hybrid retrieval" in caption.value for caption in app.caption)
    app.sidebar.selectbox[0].set_value("vector").run()
    submit(app, "Find sources")
    factory.assert_called_with(persist_dir="chroma_db", retrieval="vector")
    assert factory.call_count == 2
    app.sidebar.selectbox[0].set_value("hybrid").run()
    submit(app, "Find sources")
    assert factory.call_count == 2
    submit(app, "Reload index")
    submit(app, "Find sources")
    assert factory.call_count == 3
    factory.assert_called_with(persist_dir="chroma_db", retrieval="hybrid")


def test_empty_question_does_not_load_engine(ui):
    app, _, factory = ui
    app.text_area[0].set_value("   ")
    submit(app, "Find sources")
    factory.assert_not_called()
    assert app.error[0].value == "Enter a question first."


def test_missing_index_explains_how_to_ingest(ui):
    app, _, factory = ui
    factory.side_effect = NotFoundError("Collection not found")
    submit(app, "Find sources")
    assert "python ingest.py" in app.error[0].value


def test_failed_query_clears_previous_result(ui):
    app, engine, _ = ui
    submit(app, "Find sources")
    engine.answer_with_citations.side_effect = RuntimeError("Provider unavailable")
    submit(app, "Generate answer")
    assert app.error[0].value == "Query failed: Provider unavailable"
    assert not app.expander


def test_no_matches_is_visible_without_generation(ui):
    app, engine, _ = ui
    engine.retrieve.return_value = []
    submit(app, "Find sources")
    assert "No relevant documents found" in app.info[0].value
    engine.answer_with_citations.assert_not_called()


def test_reload_index_recreates_cached_engine(ui):
    app, _, factory = ui
    submit(app, "Find sources")
    submit(app, "Find sources")
    assert factory.call_count == 1
    submit(app, "Reload index")
    assert not app.expander
    submit(app, "Find sources")
    assert factory.call_count == 2


def test_upload_indexes_documents_and_reloads_engine(ui, monkeypatch, tmp_path):
    from ingest import save_uploaded_documents

    app, _, factory = ui
    submit(app, "Find sources")
    monkeypatch.setattr(
        "ingest.save_uploaded_documents",
        lambda files: save_uploaded_documents(files, str(tmp_path)),
    )
    runner = Mock(return_value=42)
    monkeypatch.setattr("ingest.ingest_documents", runner)
    app.file_uploader[0].set_value([
        ("runbook.md", b"# Runbook\n\n" + b"Check database latency and slow queries. " * 3, "text/markdown"),
    ])
    submit(app, "Upload and index")
    runner.assert_called_once_with(persist_dir="chroma_db")
    assert len(list(tmp_path.rglob("runbook.md"))) == 1
    assert "Indexed 42 chunks" in app.success[0].value
    assert not app.expander
    submit(app, "Find sources")
    assert factory.call_count == 2


def test_empty_upload_does_not_rebuild_index(ui, monkeypatch):
    app, _, _ = ui
    runner = Mock()
    monkeypatch.setattr("ingest.ingest_documents", runner)
    submit(app, "Upload and index")
    assert "Choose at least one" in app.error[0].value
    runner.assert_not_called()


def test_indexing_failure_keeps_saved_upload_and_clears_stale_results(ui, monkeypatch, tmp_path):
    from ingest import save_uploaded_documents

    app, _, _ = ui
    submit(app, "Find sources")
    monkeypatch.setattr(
        "ingest.save_uploaded_documents",
        lambda files: save_uploaded_documents(files, str(tmp_path)),
    )
    monkeypatch.setattr("ingest.ingest_documents", Mock(side_effect=RuntimeError("disk full")))
    app.file_uploader[0].set_value([
        ("runbook.md", b"# Runbook\n\n" + b"Check database latency and slow queries. " * 3, "text/markdown"),
    ])
    submit(app, "Upload and index")
    assert "Documents were saved, but indexing failed" in app.error[0].value
    assert len(list(tmp_path.rglob("runbook.md"))) == 1
    assert not app.success
    assert not app.expander
