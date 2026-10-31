"""Hybrid (BM25 + vector) retrieval with fixed embeddings and no model download."""

from unittest.mock import Mock

import chromadb
import pytest

import evals
from query import RAGEngine, bm25_scores, reciprocal_rank_fusion, tokenize

# Fixed 2-d embeddings: the query embedding is [1, 0], so cosine distance
# grows from "near" to "far".
CHUNKS = {
    "near": ("Coverage is reported per cloud and per tier.", [1.0, 0.0]),
    "middle": ("GCP coverage is currently 0.47% after onboarding began.", [0.8, 0.6]),
    "far": ("The Ops Agent is configured with a YAML file.", [0.0, 1.0]),
}


@pytest.fixture
def engine_factory(tmp_path, monkeypatch):
    client = chromadb.PersistentClient(path=str(tmp_path))
    collection = client.create_collection(
        "observability-docs", metadata={"hnsw:space": "cosine"},
    )
    collection.add(
        ids=list(CHUNKS),
        documents=[text for text, _ in CHUNKS.values()],
        embeddings=[emb for _, emb in CHUNKS.values()],
        metadatas=[{"source": f"{cid}.md", "file": f"{cid}.md", "section": "S"}
                   for cid in CHUNKS],
    )
    embedder = Mock()
    embedder.encode.return_value.tolist.return_value = [[1.0, 0.0]]
    monkeypatch.setattr("query.SentenceTransformer", Mock(return_value=embedder))
    return lambda mode: RAGEngine(str(tmp_path), retrieval=mode)


def test_tokenize_keeps_exact_terms_whole():
    assert tokenize("GCP is at 0.47% for `cost-centre`; last_updated.") == [
        "gcp", "is", "at", "0.47%", "for", "cost-centre", "last_updated",
    ]


def test_bm25_prefers_rare_matching_terms():
    docs = [tokenize("ops agent yaml"), tokenize("azure monitor agent"), tokenize("tags")]
    scores = bm25_scores("Ops Agent", docs)
    assert scores[0] > scores[1] > 0
    assert scores[2] == 0
    assert bm25_scores("anything", []) == []


def test_rrf_rewards_items_ranked_well_in_both_lists():
    assert reciprocal_rank_fusion([[0, 1, 2], [1]]) == [1, 0, 2]
    # Ties fall back to the first ranking's order.
    assert reciprocal_rank_fusion([[0, 1], [1, 0]]) == [0, 1]


def test_hybrid_promotes_keyword_match_within_cutoff(engine_factory):
    query = "0.47%"
    vector = [c["id"] for c in engine_factory("vector").retrieve(query, max_distance=None)]
    hybrid = engine_factory("hybrid").retrieve(query, max_distance=None)
    assert vector == ["near", "middle", "far"]
    assert [c["id"] for c in hybrid] == ["middle", "near", "far"]
    assert hybrid[0]["source"] == "middle.md"
    assert hybrid[0]["distance"] == pytest.approx(0.2)


def test_hybrid_still_applies_distance_cutoff_and_n_results(engine_factory):
    engine = engine_factory("hybrid")
    # "far" is the only keyword match but sits beyond the cutoff.
    assert [c["id"] for c in engine.retrieve("Ops Agent YAML", max_distance=0.6)] == [
        "near", "middle",
    ]
    assert len(engine.retrieve("Ops Agent YAML", n_results=1, max_distance=None)) == 1


def test_hybrid_without_keyword_overlap_matches_vector(engine_factory):
    query = "unrelated words"
    hybrid = engine_factory("hybrid").retrieve(query)
    vector = engine_factory("vector").retrieve(query)
    assert [c["id"] for c in hybrid] == [c["id"] for c in vector] == ["near", "middle"]
    # Chroma stores float32 embeddings, so distances agree only approximately.
    assert [c["distance"] for c in hybrid] == pytest.approx([c["distance"] for c in vector])


def test_unknown_retrieval_mode_is_rejected():
    with pytest.raises(ValueError, match="Unknown retrieval mode 'bm25'"):
        RAGEngine("unused", retrieval="bm25")


def test_evals_cli_passes_retrieval_mode(monkeypatch):
    monkeypatch.setattr("sys.argv", ["evals.py", "--retrieval", "hybrid"])
    factory = Mock()
    monkeypatch.setattr(evals, "RAGEngine", factory)
    monkeypatch.setattr(evals, "run_retrieval_evals", lambda *args, **kwargs: (1, 0))
    with pytest.raises(SystemExit) as exc:
        evals.main()
    assert exc.value.code == 0
    factory.assert_called_once_with(persist_dir="chroma_db", retrieval="hybrid")
