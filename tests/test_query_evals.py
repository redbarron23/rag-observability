"""Unit tests for provider validation and eval fact matching (no API calls)."""

import pytest

from evals import contains_fact
from query import RAGEngine


def test_fact_must_start_at_word_boundary():
    assert contains_fact("Target is 85% within 4 months.", "4 months")
    assert not contains_fact("Target is 85% within 14 months.", "4 months")
    assert not contains_fact("Target is 85%.", "5%")


def test_fact_match_is_case_insensitive_and_allows_suffixes():
    assert contains_fact("Owner and Cost-Centre tags", "cost-centre")
    assert contains_fact("Each service defines SLOs", "SLO")


def test_unknown_provider_is_rejected(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "antropic")
    engine = object.__new__(RAGEngine)  # skip loading Chroma and the embedder
    with pytest.raises(ValueError, match="Unknown LLM provider 'antropic'"):
        engine._call_llm("question", "context")
