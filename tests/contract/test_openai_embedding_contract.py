from __future__ import annotations

import json

import httpx
import pytest

from app.shared.contracts.embedding import RerankDocument
from app.storage.bm25 import Bm25Encoder
from app.storage.embedding import EmbeddingValidationError
from app.storage.embedding_openai import OpenAICompatibleEmbedding

DIMENSION = 8


def _client(handler) -> OpenAICompatibleEmbedding:
    return OpenAICompatibleEmbedding(
        base_url="https://gateway.test/api/v1",
        token="key",
        dense_model="qwen/qwen3-embedding-8b",
        reranker_model="qwen/qwen3-reranker-4b",
        dense_dimension=DIMENSION,
        sparse_encoder=Bm25Encoder(),
        transport=httpx.MockTransport(handler),
    )


def _embedding(seed: float) -> list[float]:
    return [seed, 1.0] + [0.0] * (DIMENSION - 2)


async def test_dense_sends_the_openai_shape_and_restores_input_order() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": _embedding(2.0)},
                    {"index": 0, "embedding": _embedding(1.0)},
                ]
            },
        )

    vectors = await _client(handler).dense(["first", "second"])

    assert seen["path"] == "/api/v1/embeddings"
    assert seen["body"]["input"] == ["first", "second"]
    assert seen["body"]["dimensions"] == DIMENSION
    assert vectors[0][0] < vectors[1][0]
    assert sum(v * v for v in vectors[0]) == pytest.approx(1.0)


async def test_dense_rejects_a_wrong_dimension() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0, 2.0]}]})

    with pytest.raises(EmbeddingValidationError):
        await _client(handler).dense(["only one"])


async def test_dense_rejects_a_short_response() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": _embedding(1.0)}]})

    with pytest.raises(EmbeddingValidationError):
        await _client(handler).dense(["one", "two"])


async def test_rerank_maps_indices_back_to_document_ids() -> None:
    documents = [RerankDocument(id=f"chunk-{i}", text=f"text {i}") for i in range(3)]

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert request.url.path == "/api/v1/rerank"
        assert body["documents"] == ["text 0", "text 1", "text 2"]
        return httpx.Response(
            200,
            json={"results": [{"index": 2, "relevance_score": 0.9}, {"index": 0, "score": 0.4}]},
        )

    response = await _client(handler).rerank("q", documents, top_n=2)

    assert [(r.id, r.score) for r in response.results] == [("chunk-2", 0.9), ("chunk-0", 0.4)]


async def test_rerank_rejects_an_index_outside_the_supplied_set() -> None:
    documents = [RerankDocument(id="chunk-0", text="text")]

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [{"index": 7, "relevance_score": 1.0}]})

    with pytest.raises(EmbeddingValidationError):
        await _client(handler).rerank("q", documents, top_n=1)


async def test_sparse_never_leaves_the_process() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:  # pragma: no cover - must not run
        raise AssertionError("sparse must not call the gateway")

    vectors = await _client(handler).sparse(["отпуск и командировки"])
    assert vectors[0].indices and len(vectors[0].indices) == len(vectors[0].values)
