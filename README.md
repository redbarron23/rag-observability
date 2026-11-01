# rag-observability — Cited Answers from Observability Docs

[![CI](https://github.com/redbarron23/rag-observability/actions/workflows/ci.yml/badge.svg)](https://github.com/redbarron23/rag-observability/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Find answers in your observability documentation and see the supporting sources.**
This local retrieval-augmented generation (RAG) pipeline indexes Markdown documents, retrieves relevant
sections, and asks Claude or DeepSeek to answer with document and section citations.

![Answering an observability question with a cited source](docs/demo.gif)

## Quick start

Requirements: Python **3.10+**. Embedding runs locally on CPU; the embedding model downloads on first use.

```bash
git clone https://github.com/redbarron23/rag-observability.git
cd rag-observability
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python ingest.py
python evals.py
```

This indexes the bundled fictional corpus and runs retrieval checks without a provider key.
To generate an answer, configure a provider:

```bash
export ANTHROPIC_API_KEY=your-key
python query.py "What is the coverage target for Tier 1 production resources?"
```

For DeepSeek, set `DEEPSEEK_API_KEY` and pass `--provider deepseek`.
Generated answers use a hosted LLM and may incur provider charges. See [the guide](docs/guide.md#quick-start).

## Local web interface

```bash
python -m pip install -r requirements-ui.txt
python ingest.py
python -m streamlit run app.py --server.address 127.0.0.1 --server.fileWatcherType none
```

Open the local URL printed in the terminal. **Find sources** needs no provider key: inspect
ranked chunks, source documents, sections, and cosine distances. Choose the retrieval mode
(`vector`, the default, or `hybrid`; see [hybrid retrieval](docs/guide.md#hybrid-retrieval-opt-in)),
chunk count and distance cutoff in the sidebar. **Generate answer** uses the same `LLM_PROVIDER`,
`LLM_MODEL`, and provider API-key environment variables as the CLI. Set them before starting
Streamlit. With Claude, a **Quoted citations** expander under the answer lists the exact text each
citation quotes; DeepSeek answers carry only the citations written in the answer text.
Each question is independent; earlier questions are not sent as conversation history.

To add documents, use **Add documents** in the sidebar: choose one or more `.md` files,
then click **Upload and index**. Files must be UTF-8 text with at least 50 characters of
usable content, up to 5 MB each. Uploads are saved under `data/uploads/` (excluded from Git).
The selected index is updated from all documents in `data/`, including existing uploads;
only new or changed files are embedded, and the cached engine is cleared automatically. Different contents with the same filename
are kept separately; identical re-uploads reuse the saved file. See [data handling](docs/guide.md#data-handling)
for command-line ingestion, document structure, and rebuild limitations.

Re-running `python ingest.py` embeds only added or changed files, drops deleted ones, and
does nothing when sources are unchanged; `python ingest.py --rebuild` re-embeds everything.
Upgrading `sentence-transformers` or the cached model triggers a full re-embed on the next ingest
automatically. See [incremental ingestion](docs/guide.md#incremental-ingestion).
`python ingest.py --prune` deletes superseded index collections, keeping the active and previous
ones for rollback. See [pruning old collections](docs/guide.md#pruning-old-collections).
After re-ingesting documents, click **Reload index** to reload the cached engine.
Use `python ingest.py --check` to detect changed local source documents or an embedding-version mismatch; verbose CLI queries
show the indexed timestamp and corpus hash. See [corpus provenance](docs/guide.md#indexed-corpus-provenance-and-freshness).
The launch command disables file watching to avoid inspecting the model libraries; restart
the app after editing its Python code.
The interface is intended for local use and has no authentication.

## How it works

```mermaid
flowchart LR
    D[Markdown documents] --> C[Heading-aware chunks]
    C --> E[Local embeddings]
    E --> V[(ChromaDB)]
    Q[Question] --> R[Retrieve relevant sections]
    V --> R
    R --> L[Claude or DeepSeek]
    L --> A[Answer with source citations]
```

Ingestion preserves heading breadcrumbs. Retrieval searches a local vector database; generation
receives the selected chunks. Claude receives each chunk as a document with native citations
enabled, so its citations name the exact cited text and the chunk it came from; the CLI lists
them after the answer. DeepSeek is prompted to write a citation for each factual claim.
Retrieval and answer quality are evaluated separately.

## Quality and limits

- The included corpus contains **eight fictional observability documents**, not customer material.
- Evals cover seventeen answerable questions (expected sources, supporting sections, facts, and
  citations) and four the corpus can't answer (the model must decline). With Claude Sonnet 5.5 all
  21 pass in both retrieval modes ([latest results](docs/guide.md#latest-results)). Passing on this small corpus
  does not establish quality on an unseen estate; adapt the eval cases for your own documents.
- Citations come from the LLM: Claude's native citations quote the cited text, and DeepSeek's are
  written by the model. Check that the cited section actually supports the answer.
- Chunks with a cosine distance above `--max-distance` (default 0.6, measured on the bundled
  corpus) are dropped, except that a chunk up to 0.1 further away is kept when it contains an
  exact term from the question (`last_updated`, `0.47%`, "Ops Agent"). Re-measure the cutoff
  on your own corpus rather than assuming it carries over.
- Embeddings and the vector store stay local. Relevant retrieved text is sent to the configured
  provider when generating an answer; use documents approved for that destination.
- `--retrieval hybrid` (on `query.py` and `evals.py`) fuses vector ranking with BM25 keyword
  ranking for exact terms such as `0.47%` or `Ops Agent`. Vector-only stays the default; see
  [hybrid retrieval](docs/guide.md#hybrid-retrieval-opt-in) for the eval comparison.

## Documentation

| Topic | Guide |
| --- | --- |
| Full walkthrough and CLI options | [Usage guide](docs/guide.md#usage-examples) |
| Pipeline and module map | [Architecture](docs/guide.md#the-rag-pipeline-detailed) |
| Heading-aware chunking | [Chunking strategy](docs/guide.md#chunking-strategy) |
| Retrieval and generation | [Retrieval](docs/guide.md#retrieval-how-it-works), [generation](docs/guide.md#generation-the-cited-answer) |
| Source attribution, fact and citation checks | [Evaluation](docs/guide.md#evaluation-strategy) |
| Use your own corpus | [Data handling](docs/guide.md#data-handling) |

## Development

```bash
python -m pip install -r requirements-dev.txt
python -m ruff check .
python -m pytest tests/ -q
python ingest.py
python evals.py
```

CI runs Ruff linting, unit tests, ingestion, and retrieval evals. Generation evals (`python evals.py --with-llm`)
are opt-in and call the configured provider. A separate workflow runs them against Claude weekly and on
manual dispatch, in both retrieval modes (about $0.12 per mode per run), and never on pull requests. It needs
an `ANTHROPIC_API_KEY` repository secret (`gh secret set ANTHROPIC_API_KEY`); without it the run is skipped.
See [scheduled LLM evals](docs/guide.md#scheduled-llm-evals).

Follow the [open-source roadmap](https://github.com/users/redbarron23/projects/2).

## License

MIT. See [LICENSE](LICENSE).
