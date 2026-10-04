#!/usr/bin/env python3
"""Query the observability RAG system.

Usage
-----
    python query.py "What are the coverage targets for production Azure resources?"
    python query.py --n-results 5 "What is the observability framework scoring model?"
    python query.py --verbose "How does GCP metric collection differ from Azure?"

This retrieves the top-k relevant chunks from Chroma, then asks an LLM
to answer the question using ONLY those chunks as sources (cited).

Default provider is DeepSeek. Set LLM_PROVIDER=anthropic to use Claude.
"""

import argparse
import os
import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

# ── Defaults ───────────────────────────────────────────────────────────

PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "observability-docs"
N_RESULTS = 6
EMBED_MODEL = "all-MiniLM-L6-v2"

DEFAULT_PROVIDER = "deepseek"
DEFAULT_MODEL = "deepseek-chat"

PROVIDER_DEFAULT_MODELS = {
    "deepseek": "deepseek-chat",
    "anthropic": "claude-sonnet-4-6",
}

SYSTEM_PROMPT = """You are an observability domain expert assistant.

You answer questions based ONLY on the provided context documents.
If the context does not contain enough information to answer, say so clearly.

For every factual claim, cite the source document and section in parentheses.
Be concise and precise — this is technical infrastructure knowledge.

Example:
    "The production Tier 1 coverage target is 90% within 6 months and 100%
    within 12 months (source: coverage-targets.md, section 'Tier 1')."

Do NOT use any external knowledge. Stick strictly to what the retrieved
chunks contain."""


# ── RAG engine ─────────────────────────────────────────────────────────

class RAGEngine:
    """Retrieve-and-generate over observability docs."""

    def __init__(self, persist_dir: str = PERSIST_DIR):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_collection(COLLECTION_NAME)
        self.embedder = SentenceTransformer(EMBED_MODEL)

    def retrieve(self, query: str, n_results: int = N_RESULTS) -> list[dict]:
        """Embed the query and retrieve the top-k matching chunks."""
        query_embedding = self.embedder.encode([query]).tolist()[0]

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
        return chunks

    def answer(self, query: str, n_results: int = N_RESULTS,
               verbose: bool = False) -> tuple[str, list[dict]]:
        """Retrieve relevant chunks and generate an answer via the configured LLM."""
        chunks = self.retrieve(query, n_results=n_results)

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
                print(f"\n--- Chunk {i+1} (source: {c['source']}, "
                      f"dist: {c['distance']:.4f}) ---")
                print(c["text"][:400])
                if len(c["text"]) > 400:
                    print("... (truncated)")

        # Call LLM
        answer = self._call_llm(query, context)
        return answer, chunks

    def _call_llm(self, query: str, context: str) -> str:
        """Send the prompt to the configured LLM provider."""
        provider = os.environ.get("LLM_PROVIDER", DEFAULT_PROVIDER)
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
            response = client.messages.create(
                model=model,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_content}],
            )
            return response.content[0].text
        else:  # deepseek (OpenAI-compatible)
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
    parser.add_argument("--provider", default=None,
                        help="LLM provider (anthropic or deepseek). Overrides LLM_PROVIDER env var.")
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

    engine = RAGEngine(persist_dir=args.persist_dir)

    if args.query:
        answer, chunks = engine.answer(
            args.query,
            n_results=args.n_results,
            verbose=args.verbose,
        )
        print(f"\nAnswer:\n{answer}")
    else:
        # Interactive REPL
        provider = os.environ.get("LLM_PROVIDER", DEFAULT_PROVIDER)
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
            answer, chunks = engine.answer(
                q, n_results=args.n_results, verbose=args.verbose,
            )
            print(f"A: {answer}\n")


if __name__ == "__main__":
    main()
