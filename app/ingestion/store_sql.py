"""PostgreSQL-backed ``IngestionStore`` (spec sections 8, 9, 11).

Finalization (READY + job COMPLETED + completed outbox event + inbox record + audit
event) commits in a single transaction so a document is never READY without its
durable side effects (invariant 7, spec step 13-14).
"""

from __future__ import annotations

from app.ingestion.ports import (
    DocumentState,
    FinalizeData,
    IndexConfigState,
)
from app.shared.enums import DocumentStatus, EventType, JobStatus
from app.shared.errors import ErrorCode, UpstreamError
from app.shared.time import utcnow
from app.storage.postgres.engine import Database
from app.storage.postgres.repositories import (
    DocumentRepository,
    EventRepository,
    InboxRepository,
    IndexConfigRepository,
    IngestionJobRepository,
    OutboxRepository,
)
from app.worker_support.mapping import document_state_from


class SqlIngestionStore:
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

    async def get_active_index_config(self) -> IndexConfigState:
        async with self._db.session() as session:
            cfg = await IndexConfigRepository(session).get_active_or_raise()
            return IndexConfigState(
                version=cfg.version,
                dense_model=cfg.dense_model,
                dense_dimension=cfg.dense_dimension,
                sparse_model=cfg.sparse_model,
                qdrant_collection=cfg.qdrant_collection,
            )

    async def find_ready_by_checksum(self, checksum: str, document_version: int) -> str | None:
        async with self._db.session() as session:
            doc = await DocumentRepository(session).find_indexed_by_checksum(
                checksum, document_version
            )
            return doc.id if doc else None

    async def begin_processing(
        self, job_id: str, document_id: str, *, owner: str, lease_ttl_seconds: int
    ) -> None:
        async with self._db.begin() as session:
            jobs = IngestionJobRepository(session)
            job = await jobs.acquire_lease(job_id, owner=owner, lease_ttl_seconds=lease_ttl_seconds)
            if job is None:
                raise UpstreamError(
                    "Job is leased by another live worker",
                    code=ErrorCode.QUEUE_TIMEOUT,
                    retryable=True,
                )
            docs = DocumentRepository(session)
            doc = await docs.get_for_update(document_id)
            await docs.transition(
                doc,
                allowed_from={
                    DocumentStatus.QUEUED,
                    DocumentStatus.PROCESSING,
                    DocumentStatus.UPLOADED,
                    DocumentStatus.FAILED,
                },
                to=DocumentStatus.PROCESSING,
            )
            job.status = JobStatus.PROCESSING.value

    async def set_stage(self, job_id: str, *, stage: str, progress: int) -> None:
        async with self._db.begin() as session:
            job = await IngestionJobRepository(session).get_for_update(job_id)
            job.stage = stage
            job.progress = progress

    async def heartbeat(self, job_id: str, *, owner: str, lease_ttl_seconds: int) -> bool:
        async with self._db.begin() as session:
            return await IngestionJobRepository(session).heartbeat(
                job_id, owner=owner, lease_ttl_seconds=lease_ttl_seconds
            )

    async def cancel_requested(self, job_id: str) -> bool:
        async with self._db.session() as session:
            job = await IngestionJobRepository(session).get_or_raise(job_id)
            return job.cancel_requested_at is not None

    async def persist_markdown_key(self, document_id: str, key: str) -> None:
        async with self._db.begin() as session:
            doc = await DocumentRepository(session).get_for_update(document_id)
            doc.markdown_object_key = key

    async def finalize_ready(
        self, *, document_id: str, job_id: str, consumer: str, event_id: str, data: FinalizeData
    ) -> None:
        async with self._db.begin() as session:
            docs = DocumentRepository(session)
            doc = await docs.get_for_update(document_id)
            await docs.transition(
                doc,
                allowed_from={DocumentStatus.PROCESSING, DocumentStatus.READY},
                to=DocumentStatus.READY,
            )
            doc.chunk_count = data.chunk_count
            doc.index_version = data.index_version
            doc.embedding_model = data.embedding_model
            doc.parser_version = data.parser_version
            doc.chunker_version = data.chunker_version
            doc.markdown_object_key = data.markdown_object_key
            doc.checksum = data.checksum
            doc.indexed_at = utcnow()
            doc.error_code = None
            doc.error_message = None

            job = await IngestionJobRepository(session).get_for_update(job_id)
            job.status = JobStatus.COMPLETED.value
            job.progress = 100

            await OutboxRepository(session).add(
                event_id=data.completed_event["event_id"],
                event_type=EventType.DOCUMENT_INGESTION_COMPLETED.value,
                aggregate_id=document_id,
                payload=data.completed_event,
                routing_key=self._events_rk,
            )
            await EventRepository(session).append(
                document_id=document_id,
                event_type=EventType.DOCUMENT_INGESTION_COMPLETED.value,
                payload={"chunk_count": data.chunk_count, "index_version": data.index_version},
                trace_id=data.completed_event.get("trace_id"),
            )
            await InboxRepository(session).record(consumer, event_id)

    async def complete_idempotent(self, *, job_id: str, consumer: str, event_id: str) -> None:
        async with self._db.begin() as session:
            job = await IngestionJobRepository(session).get_for_update(job_id)
            job.status = JobStatus.COMPLETED.value
            job.progress = 100
            inbox = InboxRepository(session)
            if not await inbox.seen(consumer, event_id):
                await inbox.record(consumer, event_id)

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
            job = await IngestionJobRepository(session).get_for_update(job_id)
            job.stage = stage
            job.attempt = attempt
            job.error_code = error_code
            job.error_message = error_message

            if set_document_failed:
                # Terminal failure: FAILED document, but keep any Markdown artifact so a
                # future retry can reuse it (compensation, spec section 11).
                docs = DocumentRepository(session)
                doc = await docs.get_for_update(document_id)
                if DocumentStatus(doc.status) not in (
                    DocumentStatus.DELETED,
                    DocumentStatus.DELETING,
                ):
                    doc.status = DocumentStatus.FAILED.value
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
                        event_type=EventType.DOCUMENT_INGESTION_FAILED.value,
                        aggregate_id=document_id,
                        payload=failed_event,
                        routing_key=self._events_rk,
                    )
                    await EventRepository(session).append(
                        document_id=document_id,
                        event_type=EventType.DOCUMENT_INGESTION_FAILED.value,
                        payload={"stage": stage, "error_code": error_code},
                        trace_id=failed_event.get("trace_id"),
                    )
                inbox = InboxRepository(session)
                if not await inbox.seen(consumer, event_id):
                    await inbox.record(consumer, event_id)
            else:
                job.status = JobStatus.RETRY_WAIT.value

    async def record_retry(self, job_id: str, *, attempt: int, available_in: float) -> None:
        from datetime import timedelta

        async with self._db.begin() as session:
            job = await IngestionJobRepository(session).get_for_update(job_id)
            job.status = JobStatus.RETRY_WAIT.value
            job.attempt = attempt
            job.available_at = utcnow() + timedelta(seconds=available_in)
