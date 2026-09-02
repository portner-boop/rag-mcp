from __future__ import annotations

from app.shared.contracts.mcp import (
    CancelJobInput,
    CancelJobOutput,
    DeleteDocumentInput,
    DeleteDocumentOutput,
    ReindexDocumentInput,
    ReindexDocumentOutput,
)
from app.shared.contracts.queue import DocumentDeletionRequested, DocumentReindexRequested
from app.shared.enums import DocumentStatus, EventType, JobStatus
from app.shared.errors import IdempotencyConflictError, InvalidStateError, ValidationError
from app.shared.ids import new_uuid
from app.shared.time import to_rfc3339, utcnow
from app.shared.time import utcnow as _utcnow
from app.shared.trace import current_trace_id
from app.storage.postgres.engine import Database
from app.storage.postgres.repositories import (
    DeletionJobRepository,
    DocumentRepository,
    EventRepository,
    IndexConfigRepository,
    IngestionJobRepository,
    OutboxRepository,
    ReindexJobRepository,
)

_DELETABLE_FROM = {
    DocumentStatus.UPLOADED,
    DocumentStatus.FAILED,
    DocumentStatus.READY,
    DocumentStatus.DELETE_FAILED,
}
_REINDEXABLE_FROM = {DocumentStatus.UPLOADED, DocumentStatus.FAILED, DocumentStatus.READY}


class LifecycleService:
    def __init__(self, *, database: Database, settings) -> None:
        self._db = database
        self._settings = settings

    async def delete_document(self, payload: DeleteDocumentInput) -> DeleteDocumentOutput:
        async with self._db.begin() as session:
            docs = DocumentRepository(session)
            jobs = DeletionJobRepository(session)
            outbox = OutboxRepository(session)
            events = EventRepository(session)

            existing = await jobs.get_by_idempotency_key(payload.idempotency_key)
            if existing is not None:
                if existing.document_id != payload.document_id:
                    raise IdempotencyConflictError(
                        "Idempotency key reused with a different document"
                    )
                return DeleteDocumentOutput(
                    document_id=existing.document_id,
                    job_id=existing.id,
                    status=DocumentStatus.DELETING.value,
                )

            doc = await docs.get_for_update(payload.document_id)
            status = DocumentStatus(doc.status)

            if status in (DocumentStatus.DELETED, DocumentStatus.DELETING):
                active = await jobs.find_active_for_document(doc.id)
                return DeleteDocumentOutput(
                    document_id=doc.id,
                    job_id=active.id if active else "",
                    status=status.value,
                )
            if status not in _DELETABLE_FROM:
                raise InvalidStateError(
                    f"Cannot delete a document in state {status.value}",
                    details={"status": status.value},
                )

            await docs.transition(doc, allowed_from=_DELETABLE_FROM, to=DocumentStatus.DELETING)
            doc.error_code = None
            doc.error_message = None

            job = await jobs.create(
                document_id=doc.id,
                idempotency_key=payload.idempotency_key,
                requested_by=payload.requested_by,
                max_attempts=self._settings.max_attempts,
            )
            trace_id = current_trace_id()
            event = DocumentDeletionRequested(
                event_id=new_uuid(),
                occurred_at=to_rfc3339(utcnow()),
                domain=self._settings.domain_id,
                document_id=doc.id,
                job_id=job.id,
                attempt=0,
                trace_id=trace_id,
                requested_by=payload.requested_by,
            )
            await outbox.add(
                event_id=event.event_id,
                event_type=EventType.DOCUMENT_DELETION_REQUESTED.value,
                aggregate_id=doc.id,
                payload=event.model_dump(mode="json"),
                routing_key=self._settings.routing_key("deletion"),
            )
            await events.append(
                document_id=doc.id,
                event_type=EventType.DOCUMENT_DELETION_REQUESTED.value,
                payload={"job_id": job.id, "requested_by": payload.requested_by},
                trace_id=trace_id,
            )
            return DeleteDocumentOutput(
                document_id=doc.id, job_id=job.id, status=DocumentStatus.DELETING.value
            )

    async def reindex_document(self, payload: ReindexDocumentInput) -> ReindexDocumentOutput:
        async with self._db.begin() as session:
            docs = DocumentRepository(session)
            jobs = ReindexJobRepository(session)
            outbox = OutboxRepository(session)
            events = EventRepository(session)
            index_configs = IndexConfigRepository(session)

            target_cfg = await index_configs.get_by_version(payload.target_index_version)
            if target_cfg is None:
                raise ValidationError(
                    "Target index version has no index config",
                    details={"target_index_version": payload.target_index_version},
                )

            existing = await jobs.get_active_target(
                payload.document_id, payload.target_index_version
            )
            if existing is not None:
                return ReindexDocumentOutput(
                    document_id=existing.document_id,
                    job_id=existing.id,
                    status=JobStatus.QUEUED.value,
                    target_index_version=existing.target_index_version,
                )

            doc = await docs.get_for_update(payload.document_id)
            status = DocumentStatus(doc.status)
            if status not in _REINDEXABLE_FROM:
                raise InvalidStateError(
                    f"Cannot reindex a document in state {status.value}",
                    details={"status": status.value},
                )
            source_version = doc.index_version

            await docs.transition(doc, allowed_from=_REINDEXABLE_FROM, to=DocumentStatus.REINDEXING)

            idempotency_key = f"reindex:{doc.id}:{payload.target_index_version}"
            job = await jobs.create(
                document_id=doc.id,
                idempotency_key=idempotency_key,
                source_index_version=source_version,
                target_index_version=payload.target_index_version,
                reason=payload.reason,
                max_attempts=self._settings.max_attempts,
            )
            trace_id = current_trace_id()
            event = DocumentReindexRequested(
                event_id=new_uuid(),
                occurred_at=to_rfc3339(utcnow()),
                domain=self._settings.domain_id,
                document_id=doc.id,
                job_id=job.id,
                attempt=0,
                trace_id=trace_id,
                source_index_version=source_version,
                target_index_version=payload.target_index_version,
                reason=payload.reason,
            )
            await outbox.add(
                event_id=event.event_id,
                event_type=EventType.DOCUMENT_REINDEX_REQUESTED.value,
                aggregate_id=doc.id,
                payload=event.model_dump(mode="json"),
                routing_key=self._settings.routing_key("reindex"),
            )
            await events.append(
                document_id=doc.id,
                event_type=EventType.DOCUMENT_REINDEX_REQUESTED.value,
                payload={"job_id": job.id, "target_index_version": payload.target_index_version},
                trace_id=trace_id,
            )
            return ReindexDocumentOutput(
                document_id=doc.id,
                job_id=job.id,
                status=JobStatus.QUEUED.value,
                target_index_version=payload.target_index_version,
            )

    async def cancel_job(self, payload: CancelJobInput) -> CancelJobOutput:
        async with self._db.begin() as session:
            for repo_cls in (IngestionJobRepository, ReindexJobRepository, DeletionJobRepository):
                job = await repo_cls(session).get(payload.job_id)
                if job is not None:
                    if job.status in (
                        JobStatus.QUEUED.value,
                        JobStatus.PROCESSING.value,
                        JobStatus.RETRY_WAIT.value,
                    ):
                        job.cancel_requested_at = _utcnow()
                        return CancelJobOutput(job_id=payload.job_id, cancel_requested=True)
                    return CancelJobOutput(job_id=payload.job_id, cancel_requested=False)
            raise InvalidStateError("Job not found", details={"job_id": payload.job_id})
