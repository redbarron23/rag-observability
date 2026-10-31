"""Local HTTP access to retrieval and cited answers; run with uvicorn api:app."""

import json
import logging
import os
from contextlib import asynccontextmanager

from chromadb.errors import NotFoundError
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator

from query import MAX_DISTANCE, N_RESULTS, PERSIST_DIR, RAGEngine

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.engine = None
    try:
        app.state.engine = RAGEngine(persist_dir=os.environ.get("RAG_PERSIST_DIR", PERSIST_DIR))
    except Exception:
        logger.exception("Could not load the RAG index/model; ingest and restart the API")
    yield
    app.state.engine = None


app = FastAPI(title="Observability documentation", lifespan=lifespan)


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=4000, strict=True)
    n_results: int = Field(default=N_RESULTS, ge=1, le=20, strict=True)
    max_distance: float = Field(default=MAX_DISTANCE, ge=0, le=2, allow_inf_nan=False, strict=True)

    @field_validator("question")
    @classmethod
    def nonempty_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Enter a question first.")
        return value


def get_engine(request: Request) -> RAGEngine:
    engine = request.app.state.engine
    if engine is None:
        raise HTTPException(503, "Index/model unavailable. Run python ingest.py and restart the API.")
    return engine


def corpus_provenance(engine: RAGEngine) -> dict:
    """Expose indexed identity, without leaking the manifest's local paths."""
    try:
        manifest = json.loads((engine.collection.metadata or {}).get("corpus_manifest", "null"))
        if manifest and manifest.get("schema_version") == 1:
            return {
                "status": "available",
                "corpus_sha256": manifest["corpus_sha256"],
                "ingested_at": manifest["ingested_at"],
                "embedding_model": manifest["embedding_model"],
            }
    except (ValueError, KeyError, TypeError, AttributeError):
        pass  # Legacy or malformed metadata must not masquerade as known provenance.
    return {"status": "unavailable"}


@app.post("/retrieve")
def retrieve(query: QueryRequest, request: Request) -> dict:
    """Return supporting chunks without calling a generation provider."""
    engine = get_engine(request)
    try:
        chunks = engine.retrieve(
            query.question, n_results=query.n_results, max_distance=query.max_distance,
        )
    except Exception as exc:
        logger.exception("Documentation retrieval failed")
        raise HTTPException(503, "Retrieval unavailable. Check the index/model and restart the API.") from exc
    return {"chunks": chunks, "corpus": corpus_provenance(engine)}


@app.post("/ask")
def ask(query: QueryRequest, request: Request) -> dict:
    """Generate a cited answer using the configured provider."""
    engine = get_engine(request)
    try:
        answer, chunks, citations = engine.answer_with_citations(
            query.question, n_results=query.n_results, max_distance=query.max_distance,
        )
    except NotFoundError as exc:
        raise HTTPException(503, "Index unavailable. Run python ingest.py and restart the API.") from exc
    except Exception as exc:
        logger.exception("Documentation answer failed")
        raise HTTPException(502, "Answer generation failed. Check the server/provider configuration.") from exc
    return {
        "answer": answer, "chunks": chunks, "citations": citations,
        "corpus": corpus_provenance(engine),
    }
