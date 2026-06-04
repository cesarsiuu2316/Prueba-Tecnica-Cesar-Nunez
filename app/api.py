"""FastAPI service exposing grounded retrieval for the external n8n workflow.

Endpoints:
  GET  /health    - liveness check.
  POST /retrieve  - given a question, return the relevant corpus context, its
                    sources, whether enough context was found, and an assembled
                    Spanish grounding prompt for the n8n LLM node to generate from.

This service never calls an LLM; generation happens in the separate n8n project.
"""

from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, field_validator
from embedding_docs import EmbeddingManager
from rag_retrieval import RAGRetriever
from vector_store import VectorStore
import uvicorn


class QueryRequest(BaseModel):
    """Incoming question from the n8n webhook."""

    question: str

    @field_validator("question")
    @classmethod
    def not_blank(cls, value: str) -> str:
        """Reject empty/whitespace-only questions (returns HTTP 422)."""
        value = value.strip()
        if not value:
            raise ValueError("La pregunta no puede estar vacía.")
        return value


class ContextItem(BaseModel):
    """A single retrieved chunk."""

    text: str
    source: str
    similarity: float


class RetrieveResponse(BaseModel):
    """The payload returned to the n8n workflow."""

    question: str
    sufficient_context: bool
    contexts: list[ContextItem]
    sources: list[str]
    grounding_prompt: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the embedding model and vector store once, at startup."""
    app.state.retriever = RAGRetriever(
        vector_store=VectorStore(),
        embedding_manager=EmbeddingManager(),
    )
    yield


app = FastAPI(title="UniLink RAG Retrieval API", lifespan=lifespan)


def get_retriever(request: Request) -> RAGRetriever:
    """Return the shared retriever created at startup."""
    return request.app.state.retriever


@app.get("/health")
def health() -> dict:
    """Liveness check."""
    return {"status": "ok"}


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(
    payload: QueryRequest,
    retriever: RAGRetriever = Depends(get_retriever),
) -> RetrieveResponse:
    """Retrieve grounded context and a Spanish grounding prompt for a question."""
    try:
        result = retriever.retrieve(payload.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error de recuperación: {e}")

    sources = []
    for ctx in result["contexts"]:
        if ctx["source"] not in sources:
            sources.append(ctx["source"])

    return RetrieveResponse(
        question=result["question"],
        sufficient_context=result["sufficient_context"],
        contexts=result["contexts"],
        sources=sources,
        grounding_prompt=result["prompt"],
    )


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
