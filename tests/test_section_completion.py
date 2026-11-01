"""Split-section completion in retrieval (issue #72), with fixed embeddings."""

from unittest.mock import Mock

import chromadb
import pytest

from query import RAGEngine, chunk_index, cosine_distance

# Query embedding is [1, 0]. "policy.md" has a section split into a table
# chunk (far from the query) and a prose chunk (near it), like the tagging
# standard's Team vocabulary.
CHUNKS = [
    ("policy.md#0", "Team vocabulary\n\nLegacy spellings map to the canonical team.",
     [1.0, 0.0], "policy.md", "Team vocabulary"),
    ("policy.md#1", "Team vocabulary\n\n| team | cost centre |\n| data-platform | CC-2002 |",
     [0.0, 1.0], "policy.md", "Team vocabulary"),
    ("policy.md#2", "Enforcement\n\nPolicy denies untagged resources.",
     [0.9, 0.44], "policy.md", "Enforcement"),
    ("big.md#0", "Long\n\npart one", [0.95, 0.31], "big.md", "Long"),
    ("big.md#1", "Long\n\npart two", [0.0, 1.0], "big.md", "Long"),
    ("big.md#2", "Long\n\npart three", [0.0, 1.0], "big.md", "Long"),
    ("big.md#3", "Long\n\npart four", [0.0, 1.0], "big.md", "Long"),
]


@pytest.fixture
def engine_factory(tmp_path, monkeypatch):
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.create_collection("observability-docs",
                                          metadata={"hnsw:space": "cosine"})
    collection.add(
        ids=[c[0] for c in CHUNKS], documents=[c[1] for c in CHUNKS],
        embeddings=[c[2] for c in CHUNKS],
        metadatas=[{"source": c[3], "file": c[3], "section": c[4]} for c in CHUNKS],
    )
    embedder = Mock()
    embedder.encode.return_value.tolist.return_value = [[1.0, 0.0]]
    monkeypatch.setattr("query.SentenceTransformer", Mock(return_value=embedder))
    return lambda mode: RAGEngine(str(tmp_path), retrieval=mode)


@pytest.mark.parametrize("mode", ["vector", "hybrid"])
def test_split_section_brings_its_other_chunk(engine_factory, mode):
    chunks = engine_factory(mode).retrieve("cost centre of data-platform", n_results=1,
                                           max_distance=0.3)
    assert [c["id"] for c in chunks] == ["policy.md#0", "policy.md#1"]
    ranked, added = chunks
    assert "expanded_from" not in ranked
    assert added["expanded_from"] == "policy.md#0"
    assert added["distance"] == pytest.approx(1.0)  # beyond the cutoff, kept with its section
    assert "CC-2002" in added["text"]


@pytest.mark.parametrize("mode", ["vector", "hybrid"])
def test_expansion_keeps_ranks_and_skips_long_and_unsplit_sections(engine_factory, mode):
    chunks = engine_factory(mode).retrieve("question", n_results=3, max_distance=0.3)
    ids = [c["id"] for c in chunks]
    # Ranked hits first and unchanged; the split section's other half last.
    assert ids[:3] == ["policy.md#0", "big.md#0", "policy.md#2"]
    assert ids[3:] == ["policy.md#1"]
    # "Long" has four chunks (> SECTION_MAX_CHUNKS) and isn't expanded.
    assert not any(i.startswith("big.md#") and i != "big.md#0" for i in ids)


@pytest.mark.parametrize("mode", ["vector", "hybrid"])
def test_nothing_kept_means_nothing_added(engine_factory, mode):
    # Unanswerable questions must stay empty: expansion only follows kept chunks.
    assert engine_factory(mode).retrieve("question", max_distance=-1) == []


def test_helpers():
    assert chunk_index("reference/a.md#12") == 12
    assert chunk_index("no-index") == 0
    assert cosine_distance([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0)
    assert cosine_distance([1.0, 0.0], [2.0, 0.0]) == pytest.approx(0.0)
