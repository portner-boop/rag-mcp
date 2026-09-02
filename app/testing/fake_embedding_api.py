"""Deterministic fake Embedding API (spec section 13).

Stateless: never stores text, no Qdrant/S3 credentials. Serves dense/sparse/rerank with
deterministic vectors so ingestion + query embeddings match. Unblocks development while
the real model/token integration is a long-lead item.

Both shapes are served, so it stands in for either provider: this stack's own contract
(``/v1/embeddings/dense``, ``/v1/embeddings/sparse``, ``/v1/rerank``) and the
OpenAI/Cohere one an OpenRouter-style gateway speaks (``/embeddings``, ``/rerank``).

Run: ``fake-embedding-api``  (env: FAKE_EMBEDDING_DIMENSION, FAKE_EMBEDDING_PORT)
"""

from __future__ import annotations

import os

from fastapi import FastAPI

from app.shared.contracts.embedding import (
    DenseEmbeddingRequest,
    DenseEmbeddingResponse,
    DenseUsage,
    RerankRequest,
    RerankResponse,
    RerankResult,
    SparseEmbeddingRequest,
    SparseEmbeddingResponse,
)
from app.testing.embeddings import dense_vector, rerank_score, sparse_vector

DIMENSION = int(os.environ.get("FAKE_EMBEDDING_DIMENSION", "1024"))

app = FastAPI(title="fake-embedding-api", docs_url=None)


@app.post("/v1/embeddings/dense", response_model=DenseEmbeddingResponse)
async def dense(req: DenseEmbeddingRequest) -> DenseEmbeddingResponse:
    vectors = [dense_vector(t, DIMENSION) for t in req.texts]
    tokens = sum(len(t.split()) for t in req.texts)
    return DenseEmbeddingResponse(
        model=req.model,
        dimension=DIMENSION,
        vectors=vectors,
        usage=DenseUsage(texts=len(req.texts), tokens=tokens),
    )


@app.post("/v1/embeddings/sparse", response_model=SparseEmbeddingResponse)
async def sparse(req: SparseEmbeddingRequest) -> SparseEmbeddingResponse:
    return SparseEmbeddingResponse(model=req.model, vectors=[sparse_vector(t) for t in req.texts])


@app.post("/v1/rerank", response_model=RerankResponse)
async def rerank(req: RerankRequest) -> RerankResponse:
    scored = [RerankResult(id=d.id, score=rerank_score(req.query, d.text)) for d in req.documents]
    scored.sort(key=lambda r: -r.score)
    return RerankResponse(results=scored[: req.top_n])


# --- OpenAI / Cohere shapes (embedding_provider = "openai") -------------------------


@app.post("/embeddings")
async def openai_embeddings(body: dict) -> dict:
    texts = body.get("input") or []
    if isinstance(texts, str):
        texts = [texts]
    dimension = int(body.get("dimensions") or DIMENSION)
    return {
        "object": "list",
        "model": body.get("model", "fake"),
        "data": [
            {"object": "embedding", "index": i, "embedding": dense_vector(t, dimension)}
            for i, t in enumerate(texts)
        ],
        "usage": {"prompt_tokens": sum(len(t.split()) for t in texts), "total_tokens": 0},
    }


@app.post("/rerank")
async def openai_rerank(body: dict) -> dict:
    query = body.get("query", "")
    documents = body.get("documents") or []
    texts = [d if isinstance(d, str) else d.get("text", "") for d in documents]
    scored = sorted(
        ({"index": i, "relevance_score": rerank_score(query, t)} for i, t in enumerate(texts)),
        key=lambda r: -r["relevance_score"],
    )
    top_n = int(body.get("top_n") or len(scored))
    return {"model": body.get("model", "fake"), "results": scored[:top_n]}


@app.get("/health/live")
async def live() -> dict:
    return {"status": "live", "dimension": DIMENSION}


def main() -> None:
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("FAKE_EMBEDDING_HOST", "127.0.0.1"),
        port=int(os.environ.get("FAKE_EMBEDDING_PORT", "8000")),
        log_config=None,
    )


if __name__ == "__main__":
    main()
