"""Ports for the deletion pipeline (spec sections 7.7, 9, 11)."""

from __future__ import annotations

from typing import Protocol

from app.ingestion.ports import DocumentState


class ObjectDelete(Protocol):
    async def exists(self, key: str) -> bool: ...
    async def delete(self, key: str) -> None: ...


class VectorDelete(Protocol):
    async def delete_document_all(self, document_id: str) -> None: ...
    async def count_all_for_document(self, document_id: str) -> int: ...


class DeletionStore(Protocol):
    async def inbox_seen(self, consumer: str, event_id: str) -> bool: ...
    async def get_document(self, document_id: str) -> DocumentState: ...
    async def begin_processing(
        self, job_id: str, document_id: str, *, owner: str, lease_ttl_seconds: int
    ) -> None: ...
    async def set_stage(self, job_id: str, *, stage: str, progress: int) -> None: ...
    async def heartbeat(self, job_id: str, *, owner: str, lease_ttl_seconds: int) -> bool: ...

    async def finalize_deleted(
        self, *, document_id: str, job_id: str, consumer: str, event_id: str, deleted_event: dict
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
