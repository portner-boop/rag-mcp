"""Dense embeddings + reranking over an OpenAI-shaped HTTP API (spec section 13).

One adapter, two upstream calls:

* ``POST {base}/embeddings`` — OpenAI's embeddings shape, which OpenRouter, vLLM,
  LiteLLM and TEI all speak. ``dimensions`` is sent so the returned vector always matches
  the Qdrant collection (Qwen3 embedding models are Matryoshka-trained and honour it).
* ``POST {base}/rerank`` — the Cohere-style rerank shape OpenRouter exposes for
  cross-encoder models such as ``qwen/qwen3-reranker-4b``.

The lexical branch never leaves the process: sparse vectors come from the local BM25
encoder, and Qdrant applies IDF. Responses are validated exactly as strictly as the
custom Embedding API client validates its own contract — a wrong dimension or a
hallucinated document id is an upstream error, not silently-wrong retrieval.
"""

from __future__ import annotations

import math

import httpx

from app.shared.contracts.embedding import (
    RerankDocument,
    RerankResponse,
    RerankResult,
    SparseVector,
)
from app.shared.errors import ErrorCode, UpstreamError
from app.storage.bm25 import Bm25Encoder
from app.storage.embedding import EmbeddingValidationError


class OpenAICompatibleEmbedding:
    def __init__(
        self,
        *,
        base_url: str,
        token: str,
        dense_model: str,
        reranker_model: str,
        dense_dimension: int,
        sparse_encoder: Bm25Encoder,
        timeout: float = 60.0,
        rerank_timeout: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._dense_model = dense_model
        self._reranker_model = reranker_model
        self._dense_dimension = dense_dimension
        self._sparse = sparse_encoder
        self._timeout = timeout
        self._rerank_timeout = rerank_timeout
        self._headers = {"Authorization": f"Bearer {token}"}
        self._transport = transport  # test seam; production builds the default transport
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> OpenAICompatibleEmbedding:
        self._http()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url, headers=self._headers, transport=self._transport
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _post(self, path: str, body: dict, *, timeout: float) -> dict:
        try:
            resp = await self._http().post(path, json=body, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except httpx.TimeoutException as exc:
            raise UpstreamError(
                "Embedding service is temporarily unavailable",
                code=ErrorCode.EMBEDDING_TIMEOUT,
            ) from exc
        except httpx.HTTPError as exc:
            raise UpstreamError("Embedding request failed") from exc

    # --- dense ---------------------------------------------------------------------

    async def dense(self, texts: list[str], *, normalize: bool = True) -> list[list[float]]:
        body = {
            "model": self._dense_model,
            "input": texts,
            "encoding_format": "float",
            "dimensions": self._dense_dimension,
        }
        raw = await self._post("/embeddings", body, timeout=self._timeout)
        data = raw.get("data")
        if not isinstance(data, list) or len(data) != len(texts):
            raise EmbeddingValidationError("Dense vector count mismatch")

        # OpenAI returns the input index on every item; never trust list order.
        ordered: list[list[float]] = [[] for _ in texts]
        for item in data:
            index = item.get("index", 0)
            if not isinstance(index, int) or not 0 <= index < len(texts):
                raise EmbeddingValidationError("Dense response has an out-of-range index")
            vector = item.get("embedding")
            if not isinstance(vector, list):
                raise EmbeddingValidationError("Dense response item has no embedding")
            ordered[index] = [float(v) for v in vector]

        for vector in ordered:
            if len(vector) != self._dense_dimension:
                raise EmbeddingValidationError(
                    "Dense dimension mismatch",
                    details={"expected": self._dense_dimension, "actual": len(vector)},
                )
            if not all(math.isfinite(v) for v in vector):
                raise EmbeddingValidationError("Dense vector contains non-finite values")
        return [_l2_normalize(v) for v in ordered] if normalize else ordered

    # --- sparse --------------------------------------------------------------------

    async def sparse(self, texts: list[str]) -> list[SparseVector]:
        """Local BM25 term weights; Qdrant's `modifier: idf` supplies the IDF factor."""
        return self._sparse.encode_many(texts)

    # --- rerank --------------------------------------------------------------------

    async def rerank(
        self, query: str, documents: list[RerankDocument], *, top_n: int
    ) -> RerankResponse:
        body = {
            "model": self._reranker_model,
            "query": query,
            "documents": [d.text for d in documents],
            "top_n": min(top_n, len(documents)),
        }
        raw = await self._post("/rerank", body, timeout=self._rerank_timeout)
        results = raw.get("results")
        if not isinstance(results, list):
            raise EmbeddingValidationError("Rerank response has no results")

        scored: list[RerankResult] = []
        for item in results:
            score = item.get("relevance_score", item.get("score"))
            index = item.get("index")
            if index is None and isinstance(item.get("document"), dict):
                index = item["document"].get("index")
            if score is None or not isinstance(index, int):
                raise EmbeddingValidationError("Rerank result is missing index or score")
            if not 0 <= index < len(documents):
                raise EmbeddingValidationError("Rerank returned an index outside the supplied set")
            scored.append(RerankResult(id=documents[index].id, score=float(score)))
        return RerankResponse(results=scored)


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0.0:
        return vector
    return [v / norm for v in vector]
