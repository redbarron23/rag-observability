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
On the Anthropic path each chunk is sent as a document with Claude's native
citations enabled; DeepSeek is prompted to write its citations in the answer.
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
# Cosine distance above which a chunk is dropped. On the sample corpus the
# answerable eval questions find a relevant chunk within 0.57, and unanswerable
# questions have nothing closer than 0.65. Cosine distance tops out at 2, so
# --max-distance 2 turns the cutoff off.
MAX_DISTANCE = 0.6
# Short questions about a named thing ("What does the Ops Agent do?",
# "What is last_updated used for?") embed poorly: the right chunk ranks first
# at ~0.64 and the cutoff drops it. A chunk up to this much beyond the cutoff
# is kept when it contains an exact term from the question (see exact_terms).
EXACT_TERM_MARGIN = 0.1

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


# Names written as a run of capitalised words: "Ops Agent", "Basic Logs".
PROPER_PHRASE_RE = re.compile(r"\b[A-Z][\w-]*(?:\s+[A-Z][\w-]*)+")


def exact_terms(query: str) -> list[str]:
    """Distinctive terms in a question that a chunk can match verbatim.

    Identifiers ("last_updated", "cost-centre", "0.47%", "p95") and multi-word
    names ("Ops Agent"). Single capitalised words such as "GCP" are too common
    in the corpus to count, and the question's first word is ignored so that
    "Which Log Analytics" reduces to "Log Analytics".
    """
    terms = [t for t in TOKEN_RE.findall(query)
             if len(t) >= 3 and (re.search(r"[_.%-]", t)
                                 or (re.search(r"\d", t) and re.search(r"[a-zA-Z]", t)))]
    words = query.split(maxsplit=1)
    rest = words[1] if len(words) > 1 else ""
    terms += [" ".join(m.split()) for m in PROPER_PHRASE_RE.findall(rest)]
    return list(dict.fromkeys(terms))


def matched_exact_terms(terms: list[str], text: str) -> list[str]:
    """The terms that appear in text as whole words, ignoring case and spacing."""
    flat = " ".join(text.split())
    return [t for t in terms
            if re.search(r"(?<!\w)" + re.escape(t) + r"(?!\w)", flat, re.IGNORECASE)]


def within_cutoff(distance: float | None, max_distance: float | None,
                  terms: list[str], text: str) -> bool:
    """The distance cutoff, relaxed by EXACT_TERM_MARGIN for exact-term matches."""
    if max_distance is None:
        return True
    if distance is None:
        return False
    if distance <= max_distance:
        return True
    return (distance <= max_distance + EXACT_TERM_MARGIN
            and bool(matched_exact_terms(terms, text)))


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

# Anthropic path: chunks arrive as documents with citations enabled, so the API
# attaches structured citations and the prompt doesn't ask for written ones.
ANTHROPIC_SYSTEM_PROMPT = """You are an observability domain expert assistant.

You answer questions based ONLY on the provided documents.
If the documents do not contain enough information to answer, say so clearly.

Support every factual claim with a citation from the documents.
Be concise and precise — this is technical infrastructure knowledge.

Do NOT use any external knowledge. Stick strictly to what the documents contain."""


# ── Citation display ───────────────────────────────────────────────────

def chunk_title(chunk: dict) -> str:
    """A chunk's source plus its section, when it has one."""
    if chunk["section"]:
        return f"{chunk['source']}, section '{chunk['section']}'"
    return chunk["source"]


def citation_marker(citations: list[dict]) -> str:
    """Inline marker in the prompt-citation format, e.g.
    " (source: a.md, section 'X'; source: b.md)"; empty if there are none."""
    labels = dict.fromkeys(f"source: {chunk_title(c)}" for c in citations)
    return f" ({'; '.join(labels)})" if labels else ""


def format_citations(citations: list[dict]) -> str:
    """Numbered list of structured citations with the quoted text, for the CLI."""
    lines = []
    for i, c in enumerate(citations, start=1):
        quote = " ".join(c["cited_text"].split())
        if len(quote) > 200:
            quote = quote[:197] + "..."
        lines.append(f"[{i}] {chunk_title(c)}: \"{quote}\"")
    return "\n".join(lines)


# ── RAG engine ─────────────────────────────────────────────────────────

class RAGEngine:
    """Retrieve-and-generate over observability docs."""

    def __init__(self, persist_dir: str = PERSIST_DIR, retrieval: str = DEFAULT_RETRIEVAL,
                 embedder: SentenceTransformer | None = None):
        """Open the active collection; pass an already-loaded embedder to reuse it
        (the HTTP service keeps one model across index reloads)."""
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
        self.collection_name = name
        self.embedder = embedder if embedder is not None else SentenceTransformer(EMBED_MODEL)

    def retrieve(self, query: str, n_results: int = N_RESULTS,
                 max_distance: float | None = MAX_DISTANCE) -> list[dict]:
        """Embed the query and retrieve the top-k matching chunks.

        Chunks with a cosine distance above max_distance (i.e. less similar)
        are dropped, unless they are within EXACT_TERM_MARGIN of it and contain
        an exact term from the query; pass None to keep all top-k.
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
        terms = exact_terms(query)
        return [c for c in chunks
                if within_cutoff(c["distance"], max_distance, terms, c["text"])]

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
        few thousand chunks. The distance cutoff (with its exact-term margin)
        is applied after fusion, so it still decides whether a question is
        answerable.
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

        terms = exact_terms(query)
        chunks = []
        for i in reciprocal_rank_fusion([vector_ranking, keyword_ranking]):
            if not within_cutoff(distances[i], max_distance, terms, corpus["documents"][i]):
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
        answer, chunks, _ = self.answer_with_citations(
            query, n_results=n_results, verbose=verbose, max_distance=max_distance,
        )
        return answer, chunks

    def answer_with_citations(
        self, query: str, n_results: int = N_RESULTS, verbose: bool = False,
        max_distance: float | None = MAX_DISTANCE,
    ) -> tuple[str, list[dict], list[dict] | None]:
        """Like answer(), plus the answer's structured citations.

        Each citation is a dict with the cited chunk's "source", "section" and
        "chunk" (1-based rank) and the exact "cited_text". Citations are None
        when the provider has no native citations (DeepSeek); the answer text
        then carries the citations the model wrote.
        """
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
            return "No relevant documents found.", chunks, []

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

        answer, citations = self._call_llm(query, chunks)
        return answer, chunks, citations

    def _call_llm(self, query: str, chunks: list[dict]) -> tuple[str, list[dict] | None]:
        """Send the question and chunks to the configured LLM provider.

        Returns the answer and its structured citations (None when the
        provider has no native citations).
        """
        provider = os.environ.get("LLM_PROVIDER", DEFAULT_PROVIDER)
        if provider not in PROVIDER_DEFAULT_MODELS:
            raise ValueError(
                f"Unknown LLM provider {provider!r}; "
                f"expected one of {sorted(PROVIDER_DEFAULT_MODELS)}"
            )
        model = os.environ.get("LLM_MODEL", "")
        if not model:
            model = PROVIDER_DEFAULT_MODELS.get(provider, DEFAULT_MODEL)

        if provider == "anthropic":
            return self._call_anthropic(query, chunks, model)

        # provider == "deepseek" (OpenAI-compatible): the prompt asks for
        # citations written into the answer text.
        context = "\n\n---\n\n".join(
            f"[Chunk {i}] Source: {c['source']}"
            + (f", Section: {c['section']}" if c["section"] else "")
            + f"\n{c['text']}"
            for i, c in enumerate(chunks, start=1)
        )
        user_content = f"""Answer the following question using ONLY the context below.
Cite the source document and section for each claim you make.

Question: {query}

Context:
{context}"""

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
        return response.choices[0].message.content, None

    @staticmethod
    def _call_anthropic(query: str, chunks: list[dict],
                        model: str) -> tuple[str, list[dict]]:
        """Ask Claude with each chunk as a document with citations enabled.

        Each cited span of the answer is followed by a "(source: …, section
        '…')" marker, so it displays like a prompt-cited answer.
        """
        import anthropic
        client = anthropic.Anthropic()
        documents = [
            {
                "type": "document",
                "source": {"type": "text", "media_type": "text/plain", "data": c["text"]},
                "title": chunk_title(c),
                "citations": {"enabled": True},
            }
            for c in chunks
        ]
        question = {
            "type": "text",
            "text": f"Answer this question using ONLY the documents above.\n\nQuestion: {query}",
        }
        # Sonnet 5.5 always thinks, and thinking counts against max_tokens,
        # so leave headroom. On a safety refusal, the server retries on a
        # fallback model ("default" routing by refusal category).
        response = client.beta.messages.create(
            model=model,
            max_tokens=16000,
            system=ANTHROPIC_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": [*documents, question]}],
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if response.stop_reason == "refusal":
            return "The model declined to answer this question.", []

        # content may start with thinking blocks; the answer arrives split into
        # text blocks, and each cited block carries its own citations.
        parts: list[str] = []
        citations: list[dict] = []
        for block in response.content:
            if block.type != "text":
                continue
            block_citations = []
            for cite in getattr(block, "citations", None) or []:
                index = getattr(cite, "document_index", None)
                if index is None or not 0 <= index < len(chunks):
                    continue
                block_citations.append({
                    "source": chunks[index]["source"],
                    "section": chunks[index]["section"],
                    "chunk": index + 1,
                    "cited_text": cite.cited_text,
                })
            parts.append(block.text + citation_marker(block_citations))
            citations.extend(block_citations)
        return "".join(parts), citations


# ── CLI ────────────────────────────────────────────────────────────────

def print_answer(answer: str, citations: list[dict] | None, prefix: str) -> None:
    """Print the answer, then the quoted text behind each native citation."""
    print(f"{prefix}{answer}")
    if citations:
        print(f"\nCitations:\n{format_citations(citations)}")


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
        answer, _, citations = engine.answer_with_citations(
            args.query,
            n_results=args.n_results,
            verbose=args.verbose,
            max_distance=args.max_distance,
        )
        print_answer(answer, citations, "\nAnswer:\n")
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
            answer, _, citations = engine.answer_with_citations(
                q, n_results=args.n_results, verbose=args.verbose,
                max_distance=args.max_distance,
            )
            print_answer(answer, citations, "A: ")
            print()


if __name__ == "__main__":
    main()
