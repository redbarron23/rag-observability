"""Unit tests for document loading and chunking (no embedding model needed)."""

from ingest import chunk_text_simple, load_documents

DOC = """# Title

Intro paragraph that is long enough to be kept as part of the document.

## Tier 1

The Tier 1 target is 90%.

### Details

Nested detail paragraph.

## Tier 2

The Tier 2 target is 70%.
"""


def test_chunks_carry_heading_breadcrumb():
    chunks = chunk_text_simple(DOC, chunk_size=500)
    texts = [c["text"] for c in chunks]
    assert any(t.startswith("Title > Tier 1\n\nThe Tier 1 target") for t in texts)
    assert any(t.startswith("Title > Tier 1 > Details") for t in texts)


def test_sibling_headings_do_not_nest():
    chunks = chunk_text_simple(DOC, chunk_size=500)
    tier2 = [c for c in chunks if "Tier 2 target" in c["text"]][0]
    assert tier2["text"].startswith("Title > Tier 2\n\n")
    assert "Tier 1" not in tier2["text"]


def test_paragraphs_never_split_mid_paragraph():
    long_para = "word " * 200
    text = f"## A\n\n{long_para.strip()}\n\nshort paragraph\n"
    chunks = chunk_text_simple(text, chunk_size=100)
    assert any(long_para.strip() in c["text"] for c in chunks)


def test_small_chunk_size_creates_more_chunks():
    text = "## A\n\n" + "\n\n".join(f"paragraph number {i} " * 5 for i in range(10))
    assert len(chunk_text_simple(text, 100)) > len(chunk_text_simple(text, 5000))


def test_load_documents_skips_stubs_and_uses_relative_paths(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "real.md").write_text("# Real\n\n" + "content " * 20)
    (tmp_path / "stub.md").write_text("tiny")
    (tmp_path / "ignored.txt").write_text("x" * 200)
    docs = load_documents(str(tmp_path))
    assert [d["rel_path"] for d in docs] == ["sub/real.md"]


def test_sample_corpus_loads():
    docs = load_documents("data")
    assert len(docs) >= 8
    assert "reference/coverage-targets.md" in {d["rel_path"] for d in docs}
