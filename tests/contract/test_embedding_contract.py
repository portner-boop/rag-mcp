from __future__ import annotations

from fastapi.testclient import TestClient

from app.shared.contracts.embedding import (
    DenseEmbeddingResponse,
    RerankResponse,
    SparseEmbeddingResponse,
)
from app.testing.fake_embedding_api import DIMENSION, app

client = TestClient(app)


def test_dense_response_matches_contract() -> None:
    resp = client.post(
        "/v1/embeddings/dense", json={"model": "m", "texts": ["a", "b"], "normalize": True}
    )
    assert resp.status_code == 200
    body = DenseEmbeddingResponse.model_validate(resp.json())
    assert body.dimension == DIMENSION
    assert len(body.vectors) == 2
    assert all(len(v) == DIMENSION for v in body.vectors)
    assert body.usage.texts == 2


def test_sparse_response_lengths_match() -> None:
    resp = client.post("/v1/embeddings/sparse", json={"model": "m", "texts": ["hello world"]})
    body = SparseEmbeddingResponse.model_validate(resp.json())
    assert len(body.vectors) == 1
    v = body.vectors[0]
    assert len(v.indices) == len(v.values)


def test_rerank_returns_ids_from_supplied_set_only() -> None:
    docs = [{"id": "A", "text": "leave transfer"}, {"id": "B", "text": "remote work"}]
    resp = client.post(
        "/v1/rerank", json={"model": "m", "query": "leave", "documents": docs, "top_n": 2}
    )
    body = RerankResponse.model_validate(resp.json())
    assert {r.id for r in body.results}.issubset({"A", "B"})
    scores = [r.score for r in body.results]
    assert scores == sorted(scores, reverse=True)


def test_dense_rejects_unknown_field() -> None:
    resp = client.post("/v1/embeddings/dense", json={"model": "m", "texts": ["a"], "bogus": 1})
    assert resp.status_code == 422
