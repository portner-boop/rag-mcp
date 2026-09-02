"""PostgreSQL-backed ReindexStore with atomic cutover (spec sections 9, 12.3)."""

from __future__ import annotations

from datetime import timedelta

from app.ingestion.ports import DocumentState, IndexConfigState
from app.reindex.ports import ReindexFinalizeData
from app.shared.enums import DocumentStatus, EventType, JobStatus
from app.shared.errors import ErrorCode, InvalidStateError, UpstreamError
from app.shared.time import utcnow
from app.storage.postgres.engine import Database
from app.storage.postgres.repositories import (
    DocumentRepository,
    EventRepository,
    InboxRepository,
    IndexConfigRepository,
    OutboxRepository,
    ReindexJobRepository,
)
from app.worker_support.mapping import document_state_from


class SqlReindexStore:
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

    async def get_index_config(self, version: int) -> IndexConfigState:
        async with self._db.session() as session:
            cfg = await IndexConfigRepository(session).get_by_version(version)
            if cfg is None:
                raise InvalidStateError("Target index config missing", details={"version": version})
            return IndexConfigState(
                version=cfg.version,
                dense_model=cfg.dense_model,
                dense_dimension=cfg.dense_dimension,
                sparse_model=cfg.sparse_model,
                qdrant_collection=cfg.qdrant_collection,
                reranker_model=cfg.reranker_model,
                chunk_size_tokens=cfg.chunk_size_tokens,
                chunk_overlap_tokens=cfg.chunk_overlap_tokens,
            )

    async def begin_processing(
        self, job_id: str, document_id: str, *, owner: str, lease_ttl_seconds: int
    ) -> None:
        async with self._db.begin() as session:
            job = await ReindexJobRepository(session).acquire_lease(
                job_id, owner=owner, lease_ttl_seconds=lease_ttl_seconds
            )
            if job is None:
                raise UpstreamError(
                    "Reindex job is leased by another live worker",
                    code=ErrorCode.QUEUE_TIMEOUT,
                    retryable=True,
                )
            docs = DocumentRepository(session)
            doc = await docs.get_for_update(document_id)
            await docs.transition(
                doc, allowed_from={DocumentStatus.REINDEXING}, to=DocumentStatus.REINDEXING
            )
            job.status = JobStatus.PROCESSING.value

    async def set_stage(self, job_id: str, *, stage: str, progress: int) -> None:
        async with self._db.begin() as session:
            job = await ReindexJobRepository(session).get_for_update(job_id)
            job.stage = stage
            job.progress = progress

    async def heartbeat(self, job_id: str, *, owner: str, lease_ttl_seconds: int) -> bool:
        async with self._db.begin() as session:
            return await ReindexJobRepository(session).heartbeat(
                job_id, owner=owner, lease_ttl_seconds=lease_ttl_seconds
            )

    async def cancel_requested(self, job_id: str) -> bool:
        async with self._db.session() as session:
            job = await ReindexJobRepository(session).get_or_raise(job_id)
            return job.cancel_requested_at is not None

    async def finalize_cutover(
        self,
        *,
        document_id: str,
        job_id: str,
        consumer: str,
        event_id: str,
        data: ReindexFinalizeData,
    ) -> None:
        async with self._db.begin() as session:
            docs = DocumentRepository(session)
            doc = await docs.get_for_update(document_id)
            await docs.transition(
                doc,
                allowed_from={DocumentStatus.REINDEXING, DocumentStatus.READY},
                to=DocumentStatus.READY,
            )
            doc.index_version = data.target_index_version
            doc.chunk_count = data.chunk_count
            doc.embedding_model = data.embedding_model
            doc.parser_version = data.parser_version
            doc.chunker_version = data.chunker_version
            doc.indexed_at = utcnow()
            doc.error_code = None
            doc.error_message = None

            # Cutover: activate the target index config only now (spec 12.3).
            await IndexConfigRepository(session).activate(data.target_index_version)

            job = await ReindexJobRepository(session).get_for_update(job_id)
            job.status = JobStatus.COMPLETED.value
            job.progress = 100

            await OutboxRepository(session).add(
                event_id=data.completed_event["event_id"],
                event_type=EventType.DOCUMENT_REINDEX_COMPLETED.value,
                aggregate_id=document_id,
                payload=data.completed_event,
                routing_key=self._events_rk,
            )
            await EventRepository(session).append(
                document_id=document_id,
                event_type=EventType.DOCUMENT_REINDEX_COMPLETED.value,
                payload={"target_index_version": data.target_index_version},
                trace_id=data.completed_event.get("trace_id"),
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
            job = await ReindexJobRepository(session).get_for_update(job_id)
            job.stage = stage
            job.attempt = attempt
            job.error_code = error_code
            job.error_message = error_message
            if not set_document_failed:
                job.status = JobStatus.RETRY_WAIT.value
                return

            # Terminal: restore the document to READY on its OLD version (old version stays
            # active — the target was never activated). Failed reindex is non-destructive.
            docs = DocumentRepository(session)
            doc = await docs.get_for_update(document_id)
            if DocumentStatus(doc.status) == DocumentStatus.REINDEXING:
                doc.status = DocumentStatus.READY.value
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
                    event_type=EventType.DOCUMENT_REINDEX_FAILED.value,
                    aggregate_id=document_id,
                    payload=failed_event,
                    routing_key=self._events_rk,
                )
                await EventRepository(session).append(
                    document_id=document_id,
                    event_type=EventType.DOCUMENT_REINDEX_FAILED.value,
                    payload={"stage": stage, "error_code": error_code},
                    trace_id=failed_event.get("trace_id"),
                )
            inbox = InboxRepository(session)
            if not await inbox.seen(consumer, event_id):
                await inbox.record(consumer, event_id)

    async def record_retry(self, job_id: str, *, attempt: int, available_in: float) -> None:
        async with self._db.begin() as session:
            job = await ReindexJobRepository(session).get_for_update(job_id)
            job.status = JobStatus.RETRY_WAIT.value
            job.attempt = attempt
            job.available_at = utcnow() + timedelta(seconds=available_in)
