#!/usr/bin/env python3
"""Query the observability RAG system.

Usage
-----
    python query.py "What are the coverage targets for production Azure resources?"
    python query.py --n-results 5 "What is the observability framework scoring model?"
    python query.py --verbose "How does GCP metric collection differ from Azure?"

This retrieves the top-k relevant chunks from Chroma, then asks an LLM
to answer the question using ONLY those chunks as sources (cited).

Default provider is Anthropic (Claude). Set LLM_PROVIDER=deepseek to use DeepSeek.
"""

import argparse
import json
import math
import os
import re
from collections import Counter
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from ingest import EMBED_MODEL

# ── Defaults ───────────────────────────────────────────────────────────

PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "observability-docs"
N_RESULTS = 6
# Cosine distance above which a chunk is dropped. On the sample corpus every
# answerable eval question has a relevant chunk within 0.34, and unanswerable
# questions have nothing closer than 0.65. Cosine distance tops out at 2, so
# --max-distance 2 turns the cutoff off.
MAX_DISTANCE = 0.6

# "vector" ranks chunks by embedding similarity alone. "hybrid" fuses that
# ranking with a BM25 keyword ranking (reciprocal rank fusion), which helps
# with exact terms such as "0.47%", "Ops Agent" or tag names that MiniLM
# embeddings blur. The distance cutoff applies to both.
RETRIEVAL_MODES = ("vector", "hybrid")
DEFAULT_RETRIEVAL = "vector"
RRF_K = 60  # standard reciprocal rank fusion constant
BM25_K1 = 1.5
BM25_B = 0.75
# Keeps decimals, percentages and hyphen/underscore names as single tokens:
# "0.47%", "cost-centre", "last_updated".
TOKEN_RE = re.compile(r"\w+(?:[.\-]\w+)*%?")


def tokenize(text: str) -> list[str]:
    """Lowercased keyword tokens for BM25."""
    return TOKEN_RE.findall(text.lower())


def bm25_scores(query: str, documents: list[list[str]]) -> list[float]:
    """Okapi BM25 score of each tokenized document for the query."""
    if not documents:
        return []
    n_docs = len(documents)
    avg_len = sum(len(d) for d in documents) / n_docs or 1.0
    doc_freq = Counter(term for d in documents for term in set(d))
    query_terms = set(tokenize(query))
    scores = []
    for doc in documents:
        counts = Counter(doc)
        score = 0.0
        for term in query_terms:
            tf = counts.get(term, 0)
            if not tf:
                continue
            df = doc_freq[term]
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            norm = BM25_K1 * (1 - BM25_B + BM25_B * len(doc) / avg_len)
            score += idf * tf * (BM25_K1 + 1) / (tf + norm)
        scores.append(score)
    return scores


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = RRF_K) -> list[int]:
    """Fuse several rankings of the same items (best first) into one."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            fused[item] = fused.get(item, 0.0) + 1 / (k + rank)
    # Ties keep the order of the first ranking (the vector ranking).
    first = {item: i for i, item in enumerate(rankings[0])} if rankings else {}
    return sorted(fused, key=lambda item: (-fused[item], first.get(item, len(first))))

DEFAULT_PROVIDER = "anthropic"
DEFAULT_MODEL = "claude-sonnet-5-5"

PROVIDER_DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-5-5",
    "deepseek": "deepseek-chat",
}

SYSTEM_PROMPT = """You are an observability domain expert assistant.

You answer questions based ONLY on the provided context documents.
If the context does not contain enough information to answer, say so clearly.

For every factual claim, cite the source document and section in parentheses.
Be concise and precise — this is technical infrastructure knowledge.

Citation format:
    "<claim> (source: <document>.md, section '<Section>')."

Do NOT use any external knowledge. Stick strictly to what the retrieved
chunks contain."""


# ── RAG engine ─────────────────────────────────────────────────────────

class RAGEngine:
    """Retrieve-and-generate over observability docs."""

    def __init__(self, persist_dir: str = PERSIST_DIR, retrieval: str = DEFAULT_RETRIEVAL):
        if retrieval not in RETRIEVAL_MODES:
            raise ValueError(
                f"Unknown retrieval mode {retrieval!r}; expected one of {list(RETRIEVAL_MODES)}"
            )
        self.retrieval = retrieval
        self._corpus = None  # loaded on first hybrid query
        self.client = chromadb.PersistentClient(path=persist_dir)
        try:
            name = (Path(persist_dir) / "active-collection").read_text(encoding="utf-8")
        except FileNotFoundError:
            name = COLLECTION_NAME  # indexes created before replacement switching
        self.collection = self.client.get_collection(name)
        self.embedder = SentenceTransformer(EMBED_MODEL)

    def retrieve(self, query: str, n_results: int = N_RESULTS,
                 max_distance: float | None = MAX_DISTANCE) -> list[dict]:
        """Embed the query and retrieve the top-k matching chunks.

        Chunks with a cosine distance above max_distance (i.e. less similar)
        are dropped; pass None to keep all top-k.
        """
        query_embedding = self.embedder.encode([query]).tolist()[0]
        if self.retrieval == "hybrid":
            return self._retrieve_hybrid(query, query_embedding, n_results, max_distance)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
        )

        chunks = []
        for i in range(len(results["ids"][0])):
            meta = results["metadatas"][0][i]
            chunks.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "source": meta.get("source", "unknown"),
                "file": meta.get("file", ""),
                "section": meta.get("section", ""),
                "distance": results["distances"][0][i] if "distances" in results else None,
            })
        if max_distance is not None:
            chunks = [c for c in chunks
                      if c["distance"] is not None and c["distance"] <= max_distance]
        return chunks

    def _load_corpus(self) -> dict:
        """Every chunk with its embedding and BM25 tokens, cached per engine."""
        if self._corpus is None:
            data = self.collection.get(include=["documents", "metadatas", "embeddings"])
            self._corpus = {
                "ids": list(data["ids"]),
                "documents": list(data["documents"]),
                "metadatas": list(data["metadatas"]),
                "embeddings": [list(map(float, e)) for e in data["embeddings"]],
                "tokens": [tokenize(d) for d in data["documents"]],
            }
        return self._corpus

    def _retrieve_hybrid(self, query: str, query_embedding: list[float], n_results: int,
                         max_distance: float | None) -> list[dict]:
        """Fuse cosine and BM25 rankings over the whole collection.

        The corpus is scored in memory, which suits local document sets of a
        few thousand chunks. Chunks beyond max_distance are dropped after
        fusion, so the cutoff still decides whether a question is answerable.
        """
        corpus = self._load_corpus()
        q_norm = math.sqrt(sum(x * x for x in query_embedding)) or 1.0
        distances = []
        for emb in corpus["embeddings"]:
            dot = sum(a * b for a, b in zip(query_embedding, emb, strict=True))
            norm = math.sqrt(sum(x * x for x in emb)) or 1.0
            distances.append(1 - dot / (q_norm * norm))
        indices = range(len(distances))
        vector_ranking = sorted(indices, key=lambda i: distances[i])
        keyword = bm25_scores(query, corpus["tokens"])
        # Only chunks sharing a query term get a keyword rank.
        keyword_ranking = sorted((i for i in indices if keyword[i] > 0),
                                 key=lambda i: -keyword[i])

        chunks = []
        for i in reciprocal_rank_fusion([vector_ranking, keyword_ranking]):
            if max_distance is not None and distances[i] > max_distance:
                continue
            meta = corpus["metadatas"][i] or {}
            chunks.append({
                "id": corpus["ids"][i],
                "text": corpus["documents"][i],
                "source": meta.get("source", "unknown"),
                "file": meta.get("file", ""),
                "section": meta.get("section", ""),
                "distance": distances[i],
            })
            if len(chunks) == n_results:
                break
        return chunks

    def answer(self, query: str, n_results: int = N_RESULTS,
               verbose: bool = False,
               max_distance: float | None = MAX_DISTANCE) -> tuple[str, list[dict]]:
        """Retrieve relevant chunks and generate an answer via the configured LLM."""
        chunks = self.retrieve(query, n_results=n_results, max_distance=max_distance)

        if verbose:
            manifest = json.loads((self.collection.metadata or {}).get("corpus_manifest", "null"))
            if manifest:
                print(f"Indexed at: {manifest['ingested_at']}")
                print(f"Corpus SHA-256: {manifest['corpus_sha256']}")
                print(f"Embedding model: {manifest['embedding_model']}")
            else:
                print("Corpus provenance unavailable; re-ingest to record it.")

        if not chunks:
            return "No relevant documents found.", chunks

        # Build context from retrieved chunks
        context_parts = []
        for i, c in enumerate(chunks):
            header = f"[Chunk {i+1}] Source: {c['source']}"
            if c["section"]:
                header += f", Section: {c['section']}"
            context_parts.append(f"{header}\n{c['text']}")

        context = "\n\n---\n\n".join(context_parts)

        if verbose:
            print(f"\n{'='*60}")
            print(f"RETRIEVED CHUNKS ({len(chunks)}):")
            print(f"{'='*60}")
            for i, c in enumerate(chunks):
                dist = f"{c['distance']:.4f}" if c["distance"] is not None else "n/a"
                print(f"\n--- Chunk {i+1} (source: {c['source']}, dist: {dist}) ---")
                print(c["text"][:400])
                if len(c["text"]) > 400:
                    print("... (truncated)")

        # Call LLM
        answer = self._call_llm(query, context)
        return answer, chunks

    def _call_llm(self, query: str, context: str) -> str:
        """Send the prompt to the configured LLM provider."""
        provider = os.environ.get("LLM_PROVIDER", DEFAULT_PROVIDER)
        if provider not in PROVIDER_DEFAULT_MODELS:
            raise ValueError(
                f"Unknown LLM provider {provider!r}; "
                f"expected one of {sorted(PROVIDER_DEFAULT_MODELS)}"
            )
        model = os.environ.get("LLM_MODEL", "")
        if not model:
            model = PROVIDER_DEFAULT_MODELS.get(provider, DEFAULT_MODEL)

        user_content = f"""Answer the following question using ONLY the context below.
Cite the source document and section for each claim you make.

Question: {query}

Context:
{context}"""

        if provider == "anthropic":
            import anthropic
            client = anthropic.Anthropic()
            # Sonnet 5.5 always thinks, and thinking counts against max_tokens,
            # so leave headroom. On a safety refusal, the server retries on a
            # fallback model ("default" routing by refusal category).
            response = client.beta.messages.create(
                model=model,
                max_tokens=16000,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_content}],
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
            if response.stop_reason == "refusal":
                return "The model declined to answer this question."
            # content may start with thinking blocks; join the text blocks
            return "".join(b.text for b in response.content if b.type == "text")
        else:  # provider == "deepseek" (OpenAI-compatible)
            from openai import OpenAI
            client = OpenAI(
                base_url="https://api.deepseek.com",
                api_key=os.environ.get("DEEPSEEK_API_KEY"),
            )
            response = client.chat.completions.create(
                model=model,
                max_tokens=1024,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
            )
            return response.choices[0].message.content


# ── CLI ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query the observability RAG system",
    )
    parser.add_argument("query", nargs="?", help="Question to answer")
    parser.add_argument("--n-results", type=int, default=N_RESULTS,
                        help=f"Number of chunks to retrieve (default: {N_RESULTS})")
    parser.add_argument("--persist-dir", default=PERSIST_DIR)
    parser.add_argument("--max-distance", type=float, default=MAX_DISTANCE,
                        help="Drop chunks with cosine distance above this "
                             f"(default: {MAX_DISTANCE}; 2 disables the cutoff)")
    parser.add_argument("--retrieval", default=DEFAULT_RETRIEVAL, choices=RETRIEVAL_MODES,
                        help="vector: embedding similarity only; hybrid: fuse it with "
                             f"BM25 keyword ranking (default: {DEFAULT_RETRIEVAL})")
    parser.add_argument("--provider", default=None,
                        choices=sorted(PROVIDER_DEFAULT_MODELS),
                        help="LLM provider. Overrides LLM_PROVIDER env var.")
    parser.add_argument("--model", default=None,
                        help="Model name. Overrides LLM_MODEL env var.")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show retrieved chunks before the answer")
    args = parser.parse_args()

    # Allow CLI flags to override env vars
    if args.provider:
        os.environ["LLM_PROVIDER"] = args.provider
    if args.model:
        os.environ["LLM_MODEL"] = args.model

    provider = os.environ.get("LLM_PROVIDER", DEFAULT_PROVIDER)
    if provider not in PROVIDER_DEFAULT_MODELS:
        parser.error(f"LLM_PROVIDER={provider!r} is not one of {sorted(PROVIDER_DEFAULT_MODELS)}")

    engine = RAGEngine(persist_dir=args.persist_dir, retrieval=args.retrieval)

    if args.query:
        answer, _ = engine.answer(
            args.query,
            n_results=args.n_results,
            verbose=args.verbose,
            max_distance=args.max_distance,
        )
        print(f"\nAnswer:\n{answer}")
    else:
        # Interactive REPL
        model = os.environ.get("LLM_MODEL", "") or PROVIDER_DEFAULT_MODELS.get(provider, DEFAULT_MODEL)
        print(f"RAG observability query — Provider: {provider}, Model: {model} — type 'quit' to exit.\n")
        while True:
            try:
                q = input("Q: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if q.lower() in ("quit", "exit", "q"):
                break
            if not q:
                continue
            answer, _ = engine.answer(
                q, n_results=args.n_results, verbose=args.verbose,
                max_distance=args.max_distance,
            )
            print(f"A: {answer}\n")


if __name__ == "__main__":
    main()
