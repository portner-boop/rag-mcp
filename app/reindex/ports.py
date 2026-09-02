"""Ports for the reindex pipeline (spec sections 7.8, 12.3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.ingestion.ports import DocumentState, IndexConfigState


@dataclass
class ReindexFinalizeData:
    target_index_version: int
    source_index_version: int | None
    chunk_count: int
    embedding_model: str
    parser_version: str
    chunker_version: str
    completed_event: dict


class ReindexStore(Protocol):
    async def inbox_seen(self, consumer: str, event_id: str) -> bool: ...
    async def get_document(self, document_id: str) -> DocumentState: ...
    async def get_index_config(self, version: int) -> IndexConfigState: ...
    async def begin_processing(
        self, job_id: str, document_id: str, *, owner: str, lease_ttl_seconds: int
    ) -> None: ...
    async def set_stage(self, job_id: str, *, stage: str, progress: int) -> None: ...
    async def heartbeat(self, job_id: str, *, owner: str, lease_ttl_seconds: int) -> bool: ...
    async def cancel_requested(self, job_id: str) -> bool: ...

    async def finalize_cutover(
        self,
        *,
        document_id: str,
        job_id: str,
        consumer: str,
        event_id: str,
        data: ReindexFinalizeData,
    ) -> None: ...

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
