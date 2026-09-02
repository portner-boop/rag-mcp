"""PostgreSQL-backed DeletionStore (spec sections 8, 9, 11)."""

from __future__ import annotations

from datetime import timedelta

from app.ingestion.ports import DocumentState
from app.shared.enums import DocumentStatus, EventType, JobStatus
from app.shared.errors import ErrorCode, UpstreamError
from app.shared.time import utcnow
from app.storage.postgres.engine import Database
from app.storage.postgres.repositories import (
    DeletionJobRepository,
    DocumentRepository,
    EventRepository,
    InboxRepository,
    OutboxRepository,
)
from app.worker_support.mapping import document_state_from


class SqlDeletionStore:
    def __init__(self, *, database: Database, events_routing_key: str) -> None:
        self._db = database
        self._events_rk = events_routing_key

    async def inbox_seen(self, consumer: str, event_id: str) -> bool:
        async with self._db.session() as session:
            return await InboxRepository(session).seen(consumer, event_id)

    async def get_document(self, document_id: str) -> DocumentState:
        async with self._db.session() as session:
            doc = await DocumentRepository(session).get_or_raise(document_id)
            return document_state_from(doc)

    async def begin_processing(
        self, job_id: str, document_id: str, *, owner: str, lease_ttl_seconds: int
    ) -> None:
        async with self._db.begin() as session:
            job = await DeletionJobRepository(session).acquire_lease(
                job_id, owner=owner, lease_ttl_seconds=lease_ttl_seconds
            )
            if job is None:
                raise UpstreamError(
                    "Deletion job is leased by another live worker",
                    code=ErrorCode.QUEUE_TIMEOUT,
                    retryable=True,
                )
            docs = DocumentRepository(session)
            doc = await docs.get_for_update(document_id)
            await docs.transition(
                doc,
                allowed_from={DocumentStatus.DELETING, DocumentStatus.DELETE_FAILED},
                to=DocumentStatus.DELETING,
            )
            job.status = JobStatus.PROCESSING.value

    async def set_stage(self, job_id: str, *, stage: str, progress: int) -> None:
        async with self._db.begin() as session:
            job = await DeletionJobRepository(session).get_for_update(job_id)
            job.stage = stage
            job.progress = progress

    async def heartbeat(self, job_id: str, *, owner: str, lease_ttl_seconds: int) -> bool:
        async with self._db.begin() as session:
            return await DeletionJobRepository(session).heartbeat(
                job_id, owner=owner, lease_ttl_seconds=lease_ttl_seconds
            )

    async def finalize_deleted(
        self, *, document_id: str, job_id: str, consumer: str, event_id: str, deleted_event: dict
    ) -> None:
        async with self._db.begin() as session:
            docs = DocumentRepository(session)
            doc = await docs.get_for_update(document_id)
            await docs.transition(
                doc,
                allowed_from={
                    DocumentStatus.DELETING,
                    DocumentStatus.DELETE_FAILED,
                    DocumentStatus.DELETED,
                },
                to=DocumentStatus.DELETED,
            )
            doc.deleted_at = utcnow()
            doc.error_code = None
            doc.error_message = None

            job = await DeletionJobRepository(session).get_for_update(job_id)
            job.status = JobStatus.COMPLETED.value
            job.progress = 100

            await OutboxRepository(session).add(
                event_id=deleted_event["event_id"],
                event_type=EventType.DOCUMENT_DELETED.value,
                aggregate_id=document_id,
                payload=deleted_event,
                routing_key=self._events_rk,
            )
            await EventRepository(session).append(
                document_id=document_id,
                event_type=EventType.DOCUMENT_DELETED.value,
                payload={"job_id": job_id},
                trace_id=deleted_event.get("trace_id"),
            )
            await InboxRepository(session).record(consumer, event_id)

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
    ) -> None:
        async with self._db.begin() as session:
            job = await DeletionJobRepository(session).get_for_update(job_id)
            job.stage = stage
            job.attempt = attempt
            job.error_code = error_code
            job.error_message = error_message
            if not set_document_failed:
                job.status = JobStatus.RETRY_WAIT.value
                return

            docs = DocumentRepository(session)
            doc = await docs.get_for_update(document_id)
            if DocumentStatus(doc.status) != DocumentStatus.DELETED:
                doc.status = DocumentStatus.DELETE_FAILED.value
                doc.error_code = error_code
                doc.error_message = error_message
                doc.row_version += 1
            job.status = (
                JobStatus.DEAD_LETTER.value
                if failed_event and failed_event.get("_dead_letter")
                else JobStatus.FAILED.value
            )
            if failed_event is not None:
                failed_event.pop("_dead_letter", None)
                await OutboxRepository(session).add(
                    event_id=failed_event["event_id"],
                    event_type=EventType.DOCUMENT_DELETION_FAILED.value,
                    aggregate_id=document_id,
                    payload=failed_event,
                    routing_key=self._events_rk,
                )
                await EventRepository(session).append(
                    document_id=document_id,
                    event_type=EventType.DOCUMENT_DELETION_FAILED.value,
                    payload={"stage": stage, "error_code": error_code},
                    trace_id=failed_event.get("trace_id"),
                )
            inbox = InboxRepository(session)
            if not await inbox.seen(consumer, event_id):
                await inbox.record(consumer, event_id)

    async def record_retry(self, job_id: str, *, attempt: int, available_in: float) -> None:
        async with self._db.begin() as session:
            job = await DeletionJobRepository(session).get_for_update(job_id)
            job.status = JobStatus.RETRY_WAIT.value
            job.attempt = attempt
            job.available_at = utcnow() + timedelta(seconds=available_in)
