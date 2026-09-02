"""HTTP client for the shared stateless Embedding API (spec section 13).

Responses are validated before use: vector count must equal text count, dense vectors
must have finite values and the configured dimension, sparse indices/values lengths must
match. Used by both the server (query embeddings for search) and the worker (ingestion).
"""

from __future__ import annotations

import math

import httpx

from app.shared.contracts.embedding import (
    DenseEmbeddingRequest,
    DenseEmbeddingResponse,
    RerankDocument,
    RerankRequest,
    RerankResponse,
    SparseEmbeddingRequest,
    SparseEmbeddingResponse,
    SparseVector,
)
from app.shared.errors import DomainError, ErrorCode, UpstreamError


class EmbeddingValidationError(DomainError):
    code = ErrorCode.INVALID_DIMENSION
    http_status = 502


class EmbeddingClient:
    def __init__(
        self,
        *,
        base_url: str,
        token: str,
        dense_model: str,
        sparse_model: str,
        reranker_model: str,
        dense_dimension: int,
        timeout: float = 60.0,
        rerank_timeout: float = 30.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._dense_model = dense_model
        self._sparse_model = sparse_model
        self._reranker_model = reranker_model
        self._dense_dimension = dense_dimension
        self._timeout = timeout
        self._rerank_timeout = rerank_timeout
        self._headers = {"Authorization": f"Bearer {token}"}
        self._client: httpx.AsyncClient | None = None
        self._local_sparse = None

    async def __aenter__(self) -> EmbeddingClient:
        self._client = httpx.AsyncClient(base_url=self._base_url, headers=self._headers)
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self._base_url, headers=self._headers)
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

    async def dense(self, texts: list[str], *, normalize: bool = True) -> list[list[float]]:
        req = DenseEmbeddingRequest(model=self._dense_model, texts=texts, normalize=normalize)
        raw = await self._post("/v1/embeddings/dense", req.model_dump(), timeout=self._timeout)
        resp = DenseEmbeddingResponse.model_validate(raw)
        if len(resp.vectors) != len(texts):
            raise EmbeddingValidationError("Dense vector count mismatch")
        if resp.dimension != self._dense_dimension:
            raise EmbeddingValidationError(
                "Dense dimension mismatch",
                details={"expected": self._dense_dimension, "actual": resp.dimension},
            )
        for vec in resp.vectors:
            if len(vec) != self._dense_dimension:
                raise EmbeddingValidationError("Dense vector has wrong length")
            if not all(math.isfinite(v) for v in vec):
                raise EmbeddingValidationError("Dense vector contains non-finite values")
        return resp.vectors

    def use_local_sparse(self, encoder) -> None:
        """Serve the lexical branch from a local BM25 encoder instead of the sparse API."""
        self._local_sparse = encoder

    async def sparse(self, texts: list[str]) -> list[SparseVector]:
        if self._local_sparse is not None:
            return self._local_sparse.encode_many(texts)
        req = SparseEmbeddingRequest(model=self._sparse_model, texts=texts)
        raw = await self._post("/v1/embeddings/sparse", req.model_dump(), timeout=self._timeout)
        resp = SparseEmbeddingResponse.model_validate(raw)
        if len(resp.vectors) != len(texts):
            raise EmbeddingValidationError("Sparse vector count mismatch")
        for vec in resp.vectors:
            if len(vec.indices) != len(vec.values):
                raise EmbeddingValidationError("Sparse indices/values length mismatch")
            if not all(math.isfinite(v) for v in vec.values):
                raise EmbeddingValidationError("Sparse vector contains non-finite values")
        return resp.vectors

    async def rerank(
        self, query: str, documents: list[RerankDocument], *, top_n: int
    ) -> RerankResponse:
        req = RerankRequest(
            model=self._reranker_model, query=query, documents=documents, top_n=top_n
        )
        raw = await self._post("/v1/rerank", req.model_dump(), timeout=self._rerank_timeout)
        resp = RerankResponse.model_validate(raw)
        allowed = {d.id for d in documents}
        if any(r.id not in allowed for r in resp.results):
            raise EmbeddingValidationError("Rerank returned an id outside the supplied set")
        return resp
