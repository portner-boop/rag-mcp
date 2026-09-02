from __future__ import annotations

from pydantic import Field

from app.shared.contracts import StrictModel


class DenseEmbeddingRequest(StrictModel):
    model: str
    texts: list[str] = Field(min_length=1)
    normalize: bool = True


class DenseUsage(StrictModel):
    texts: int
    tokens: int


class DenseEmbeddingResponse(StrictModel):
    model: str
    dimension: int
    vectors: list[list[float]]
    usage: DenseUsage


class SparseEmbeddingRequest(StrictModel):
    model: str
    texts: list[str] = Field(min_length=1)


class SparseVector(StrictModel):
    indices: list[int]
    values: list[float]


class SparseEmbeddingResponse(StrictModel):
    model: str
    vectors: list[SparseVector]


class RerankDocument(StrictModel):
    id: str
    text: str


class RerankRequest(StrictModel):
    model: str
    query: str
    documents: list[RerankDocument] = Field(min_length=1)
    top_n: int = Field(ge=1)


class RerankResult(StrictModel):
    id: str
    score: float


class RerankResponse(StrictModel):
    results: list[RerankResult]
