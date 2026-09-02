"""Deletion pipeline (spec sections 7.7, 11 compensation).

Idempotently removes a document's Qdrant points and S3 objects, then marks metadata
DELETED. A missing point/object is treated as success so a retry after a partial delete
converges. The document is already excluded from search from the moment it entered
DELETING (invariant 11). Terminal failure -> DELETE_FAILED, which a retry resumes.
"""

from __future__ import annotations

import structlog

from app.deletion.ports import DeletionStore, ObjectDelete, VectorDelete
from app.shared.contracts.queue import DocumentDeleted, DocumentDeletionRequested
from app.shared.errors import DomainError, ErrorCode
from app.shared.ids import new_uuid
from app.shared.time import to_rfc3339, utcnow
from app.worker_support.result import PipelineResult

log = structlog.get_logger("deletion_pipeline")


class DeletionPipeline:
    def __init__(
        self,
        *,
        store: DeletionStore,
        object_store: ObjectDelete,
        vector_index: VectorDelete,
        consumer_name: str,
        owner: str,
        lease_ttl_seconds: int,
        domain_id: str,
    ) -> None:
        self._store = store
        self._s3 = object_store
        self._vectors = vector_index
        self._consumer = consumer_name
        self._owner = owner
        self._lease_ttl = lease_ttl_seconds
        self._domain = domain_id

    async def run(self, event: DocumentDeletionRequested) -> PipelineResult:
        document_id = event.document_id
        job_id = event.job_id or ""
        stage = "DELETING"
        if await self._store.inbox_seen(self._consumer, event.event_id):
            return PipelineResult(status="duplicate")

        try:
            doc = await self._store.get_document(document_id)
            await self._store.begin_processing(
                job_id, document_id, owner=self._owner, lease_ttl_seconds=self._lease_ttl
            )

            # Qdrant points (all versions) — idempotent.
            stage = "QDRANT_DELETE"
            await self._store.set_stage(job_id, stage=stage, progress=30)
            await self._heartbeat(job_id)
            await self._vectors.delete_document_all(document_id)

            # S3 objects — idempotent (missing = success).
            stage = "S3_DELETE"
            await self._store.set_stage(job_id, stage=stage, progress=60)
            await self._heartbeat(job_id)
            for key in (doc.original_object_key, doc.markdown_object_key):
                if key and await self._s3.exists(key):
                    await self._s3.delete(key)

            # Verify absence (spec: delete succeeds only when points/objects are absent).
            stage = "VERIFYING"
            await self._store.set_stage(job_id, stage=stage, progress=85)
            remaining = await self._vectors.count_all_for_document(document_id)
            if remaining > 0:
                raise DomainError(
                    "Points remain after deletion",
                    code=ErrorCode.MISSING_POINTS,
                    retryable=True,
                    details={"remaining": remaining},
                )
            for key in (doc.original_object_key, doc.markdown_object_key):
                if key and await self._s3.exists(key):
                    raise DomainError(
                        "Object remains after deletion",
                        code=ErrorCode.STORAGE_TIMEOUT,
                        retryable=True,
                    )

            stage = "FINALIZING"
            await self._store.set_stage(job_id, stage=stage, progress=100)
            deleted = DocumentDeleted(
                event_id=new_uuid(),
                occurred_at=to_rfc3339(utcnow()),
                domain=self._domain,
                document_id=document_id,
                job_id=job_id,
                attempt=event.attempt,
                trace_id=event.trace_id,
            )
            await self._store.finalize_deleted(
                document_id=document_id,
                job_id=job_id,
                consumer=self._consumer,
                event_id=event.event_id,
                deleted_event=deleted.model_dump(mode="json"),
            )
            log.info("deletion.completed", document_id=document_id, job_id=job_id)
            return PipelineResult(status="completed")

        except DomainError as exc:
            log.warning(
                "deletion.stage_failed",
                document_id=document_id,
                stage=stage,
                code=exc.code.value,
                retryable=exc.retryable,
            )
            return PipelineResult(
                status="failed",
                error_code=exc.code.value,
                error_message=exc.message,
                retryable=exc.retryable,
                stage=stage,
            )

    async def _heartbeat(self, job_id: str) -> None:
        await self._store.heartbeat(job_id, owner=self._owner, lease_ttl_seconds=self._lease_ttl)
