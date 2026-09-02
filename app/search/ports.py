from __future__ import annotations

from typing import Protocol

from app.ingestion.ports import IndexConfigState
from app.shared.contracts.embedding import RerankDocument, RerankResponse, SparseVector
from app.storage.qdrant import VectorHit


class SearchStore(Protocol):
    async def get_active_index_config(self) -> IndexConfigState: ...
    async def document_exists(self, document_id: str) -> bool: ...
    async def excluded_document_ids(self) -> list[str]: ...


class VectorSearch(Protocol):
    async def dense_search(
        self,
        *,
        vector: list[float],
        limit: int,
        index_version: int,
        document_ids: list[str] | None = None,
        document_version: int | None = None,
        created_from: str | None = None,
        created_to: str | None = None,
        exclude_document_ids: list[str] | None = None,
    ) -> list[VectorHit]: ...

    async def sparse_search(
        self,
        *,
        indices: list[int],
        values: list[float],
        limit: int,
        index_version: int,
        document_ids: list[str] | None = None,
        document_version: int | None = None,
        created_from: str | None = None,
        created_to: str | None = None,
        exclude_document_ids: list[str] | None = None,
    ) -> list[VectorHit]: ...


class SearchEmbedding(Protocol):
    async def dense(self, texts: list[str]) -> list[list[float]]: ...
    async def sparse(self, texts: list[str]) -> list[SparseVector]: ...
    async def rerank(
        self, query: str, documents: list[RerankDocument], *, top_n: int
    ) -> RerankResponse: ...


class QueryExpander(Protocol):
    async def expand(self, query: str, *, num_variants: int, hyde: bool) -> list[str]: ...
