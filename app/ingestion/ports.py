from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from app.shared.contracts.embedding import SparseVector


@dataclass
class DocumentState:
    document_id: str
    filename: str
    content_type: str
    size: int
    status: str
    document_version: int
    checksum: str | None
    original_object_key: str
    markdown_object_key: str | None
    index_version: int | None = None
    chunk_count: int | None = None


@dataclass
class IndexConfigState:
    version: int
    dense_model: str
    dense_dimension: int
    sparse_model: str | None
    qdrant_collection: str
    reranker_model: str | None = None
    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 64


@dataclass
class PointData:
    id: str
    dense: list[float]
    sparse_indices: list[int]
    sparse_values: list[float]
    payload: dict = field(default_factory=dict)


class ObjectStorePort(Protocol):
    async def get_bytes(self, key: str) -> bytes: ...
    async def put_bytes(self, key: str, data: bytes, *, content_type: str) -> None: ...
    async def exists(self, key: str) -> bool: ...


class VectorIndexPort(Protocol):
    async def upsert(self, points: list[PointData]) -> None: ...
    async def count_for_document(self, document_id: str, *, index_version: int) -> int: ...
    async def delete_document(self, document_id: str, *, index_version: int) -> None: ...
    async def retrieve_ids(self, ids: list[str]) -> list[str]: ...


class EmbeddingPort(Protocol):
    async def dense(self, texts: list[str]) -> list[list[float]]: ...
    async def sparse(self, texts: list[str]) -> list[SparseVector]: ...


@dataclass
class FinalizeData:
    chunk_count: int
    index_version: int
    embedding_model: str
    parser_version: str
    chunker_version: str
    markdown_object_key: str
    checksum: str
    duration_ms: int
    completed_event: dict


class IngestionStore(Protocol):
    async def inbox_seen(self, consumer: str, event_id: str) -> bool: ...
    async def get_document(self, document_id: str) -> DocumentState: ...
    async def get_active_index_config(self) -> IndexConfigState: ...
    async def find_ready_by_checksum(self, checksum: str, document_version: int) -> str | None: ...

    async def begin_processing(
        self, job_id: str, document_id: str, *, owner: str, lease_ttl_seconds: int
    ) -> None: ...

    async def set_stage(self, job_id: str, *, stage: str, progress: int) -> None: ...
    async def heartbeat(self, job_id: str, *, owner: str, lease_ttl_seconds: int) -> bool: ...
    async def cancel_requested(self, job_id: str) -> bool: ...
    async def persist_markdown_key(self, document_id: str, key: str) -> None: ...

    async def finalize_ready(
        self, *, document_id: str, job_id: str, consumer: str, event_id: str, data: FinalizeData
    ) -> None: ...

    async def complete_idempotent(self, *, job_id: str, consumer: str, event_id: str) -> None: ...

    async def mark_failed(
        self,
        *,
        document_id: str,
        job_id: str,
        stage: str,
        error_code: str,
        error_message: str,
        attempt: int,
        set_document_failed: bool,
        failed_event: dict | None,
        consumer: str,
        event_id: str,
    ) -> None: ...

    async def record_retry(self, job_id: str, *, attempt: int, available_in: float) -> None: ...
