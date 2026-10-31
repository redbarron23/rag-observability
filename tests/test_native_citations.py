"""Native citations on the Anthropic path, with mocked provider clients (no API calls)."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import query
from evals import citation_problems
from query import RAGEngine, citation_marker, format_citations

CHUNKS = [
    {"source": "reference/coverage-targets.md", "section": "Tier 1",
     "text": "The Tier 1 target is 85% within 4 months.", "distance": 0.2},
    {"source": "reference/tagging-standard.md", "section": "",
     "text": "Every resource needs an owner tag.", "distance": 0.4},
]


def text_block(text, citations=None):
    return SimpleNamespace(type="text", text=text, citations=citations)


def char_citation(index, cited_text):
    return SimpleNamespace(
        type="char_location", document_index=index, cited_text=cited_text,
        document_title="ignored", start_char_index=0, end_char_index=len(cited_text),
    )


@pytest.fixture
def anthropic_client(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    client = Mock()
    monkeypatch.setattr("anthropic.Anthropic", Mock(return_value=client))
    return client


def engine():
    return object.__new__(RAGEngine)  # skip loading Chroma and the embedder


def test_chunks_are_sent_as_cited_documents(anthropic_client):
    create = anthropic_client.beta.messages.create
    create.return_value = SimpleNamespace(stop_reason="end_turn", content=[text_block("ok")])

    engine()._call_llm("What is the Tier 1 target?", CHUNKS)

    kwargs = create.call_args.kwargs
    assert kwargs["model"] == "claude-sonnet-5-5"
    assert kwargs["system"] == query.ANTHROPIC_SYSTEM_PROMPT
    assert kwargs["fallbacks"] == "default"
    *documents, question = kwargs["messages"][0]["content"]
    assert documents == [
        {"type": "document",
         "source": {"type": "text", "media_type": "text/plain", "data": CHUNKS[0]["text"]},
         "title": "reference/coverage-targets.md, section 'Tier 1'",
         "citations": {"enabled": True}},
        {"type": "document",
         "source": {"type": "text", "media_type": "text/plain", "data": CHUNKS[1]["text"]},
         "title": "reference/tagging-standard.md",
         "citations": {"enabled": True}},
    ]
    assert question["type"] == "text"
    assert "What is the Tier 1 target?" in question["text"]


def test_native_citations_map_to_chunks_and_inline_markers(anthropic_client):
    anthropic_client.beta.messages.create.return_value = SimpleNamespace(
        stop_reason="end_turn",
        content=[
            SimpleNamespace(type="thinking", thinking=""),
            text_block("The Tier 1 target is "),
            text_block("85% within 4 months",
                       [char_citation(0, "The Tier 1 target is 85% within 4 months.")]),
            text_block(", and resources need "),
            text_block("an owner tag", [char_citation(1, "Every resource needs an owner tag."),
                                        char_citation(7, "out of range is ignored")]),
            text_block("."),
        ],
    )

    answer, citations = engine()._call_llm("q", CHUNKS)

    assert answer == (
        "The Tier 1 target is 85% within 4 months "
        "(source: reference/coverage-targets.md, section 'Tier 1'), "
        "and resources need an owner tag (source: reference/tagging-standard.md)."
    )
    assert citations == [
        {"source": "reference/coverage-targets.md", "section": "Tier 1", "chunk": 1,
         "cited_text": "The Tier 1 target is 85% within 4 months."},
        {"source": "reference/tagging-standard.md", "section": "", "chunk": 2,
         "cited_text": "Every resource needs an owner tag."},
    ]
    # The inline markers keep the existing (source: …) display and parser working
    assert citation_problems(answer, {c["source"] for c in CHUNKS},
                             ["reference/coverage-targets.md"]) == []


def test_refusal_returns_decline_with_no_citations(anthropic_client):
    anthropic_client.beta.messages.create.return_value = SimpleNamespace(
        stop_reason="refusal", content=[])
    assert engine()._call_llm("q", CHUNKS) == ("The model declined to answer this question.", [])


def test_deepseek_keeps_prompt_citations(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.delenv("LLM_MODEL", raising=False)
    client = Mock()
    client.chat.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content="85% (source: coverage-targets.md, section 'Tier 1')."))])
    monkeypatch.setattr("openai.OpenAI", Mock(return_value=client))

    answer, citations = engine()._call_llm("q", CHUNKS)

    assert citations is None
    assert answer.startswith("85%")
    messages = client.chat.completions.create.call_args.kwargs["messages"]
    assert messages[0] == {"role": "system", "content": query.SYSTEM_PROMPT}
    assert "[Chunk 1] Source: reference/coverage-targets.md, Section: Tier 1" in messages[1]["content"]
    assert "[Chunk 2] Source: reference/tagging-standard.md\n" in messages[1]["content"]


def test_answer_keeps_its_two_value_return(monkeypatch):
    eng = engine()
    monkeypatch.setattr(eng, "retrieve", lambda *a, **k: CHUNKS)
    monkeypatch.setattr(eng, "_call_llm", lambda q, chunks: ("text", [{"source": "x"}]))
    assert eng.answer("q") == ("text", CHUNKS)
    assert eng.answer_with_citations("q") == ("text", CHUNKS, [{"source": "x"}])

    monkeypatch.setattr(eng, "retrieve", lambda *a, **k: [])
    assert eng.answer_with_citations("q") == ("No relevant documents found.", [], [])


def test_eval_prefers_structured_citations():
    retrieved = {"reference/coverage-targets.md", "reference/tagging-standard.md"}
    expected = ["reference/coverage-targets.md"]
    written = "85% (source: reference/coverage-targets.md, section 'Tier 1')."

    # Structured citations decide, even when the text has a written citation
    assert citation_problems(written, retrieved, expected, citations=[]) == ["no citations"]
    wrong = [{"source": "reference/tagging-standard.md"}]
    assert citation_problems(written, retrieved, expected, citations=wrong) == [
        "doesn't cite expected sources: ['reference/coverage-targets.md']"
    ]
    right = [{"source": "reference/coverage-targets.md"}]
    assert citation_problems("85%.", retrieved, expected, citations=right) == []
    # None falls back to parsing the answer text
    assert citation_problems(written, retrieved, expected, citations=None) == []


def test_cli_citation_formatting():
    citations = [
        {"source": "a.md", "section": "Intro", "chunk": 1, "cited_text": "Line one\n  line two"},
        {"source": "b.md", "section": "", "chunk": 2, "cited_text": "x" * 250},
        {"source": "a.md", "section": "Intro", "chunk": 1, "cited_text": "again"},
    ]
    lines = format_citations(citations).splitlines()
    assert lines[0] == "[1] a.md, section 'Intro': \"Line one line two\""
    assert lines[1] == f"[2] b.md: \"{'x' * 197}...\""
    assert citation_marker(citations) == " (source: a.md, section 'Intro'; source: b.md)"
    assert citation_marker([]) == ""
