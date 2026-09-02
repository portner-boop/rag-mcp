"""Ports for the search engine (spec section 12.2).

Written against Protocols so the pipeline runs on real adapters (Postgres/Qdrant/
Embedding) and in-memory fakes for smoke/tests.
"""

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
    """Generates alternative query strings for a low-confidence retrieval (Tier 3.1).

    Returns paraphrase variants and, when ``hyde`` is set, a hypothetical answer passage
    (HyDE). Used only on the selective expansion path, so its (LLM) cost is not paid on the
    common case. An empty list means "no expansion available" — the caller degrades quietly.
    """

    async def expand(self, query: str, *, num_variants: int, hyde: bool) -> list[str]: ...
