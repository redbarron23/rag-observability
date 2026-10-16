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


@pytest.fixture
def ui(monkeypatch):
    st.cache_resource.clear()
    engine = Mock()
    engine.retrieve.return_value = [CHUNK]
    engine.answer.return_value = ("85% within 4 months (source: coverage-targets.md).", [CHUNK])
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
    factory.assert_called_once_with(persist_dir="chroma_db")
    engine.retrieve.assert_called_once_with(
        "What is the coverage target for Tier 1 production resources?",
        n_results=6, max_distance=0.6,
    )
    engine.answer.assert_not_called()
    assert "Tier 1" in app.expander[0].label
    assert "distance 0.250" in app.expander[0].label
    assert app.text[0].value == CHUNK["text"]
    app.sidebar.slider[0].set_value(3).run()
    engine.retrieve.assert_called_once()
    assert app.text[0].value == CHUNK["text"]
    assert any("top 6" in caption.value for caption in app.caption)


def test_generation_uses_engine_answer(ui):
    app, engine, _ = ui
    submit(app, "Generate answer")
    engine.answer.assert_called_once()
    engine.retrieve.assert_not_called()
    assert any("85% within 4 months" in element.value for element in app.markdown)


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
    engine.answer.side_effect = RuntimeError("Provider unavailable")
    submit(app, "Generate answer")
    assert app.error[0].value == "Query failed: Provider unavailable"
    assert not app.expander


def test_no_matches_is_visible_without_generation(ui):
    app, engine, _ = ui
    engine.retrieve.return_value = []
    submit(app, "Find sources")
    assert "No relevant documents found" in app.info[0].value
    engine.answer.assert_not_called()


def test_reload_index_recreates_cached_engine(ui):
    app, _, factory = ui
    submit(app, "Find sources")
    submit(app, "Find sources")
    assert factory.call_count == 1
    submit(app, "Reload index")
    assert not app.expander
    submit(app, "Find sources")
    assert factory.call_count == 2
