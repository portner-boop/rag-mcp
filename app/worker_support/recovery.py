"""Stale upload/job/lease recovery (spec sections 9, 11; D04 checklist).

A crashed worker leaves a job PROCESSING with an expired lease; a stale UPLOADING document
never got its object. This service, run periodically by the worker, cleans stale uploads
and re-enqueues jobs whose lease expired by re-publishing their original *Requested event
(business idempotency guarantees no double effect).
"""

from __future__ import annotations

from datetime import timedelta

import structlog

from app.observability import metrics
from app.shared.contracts.queue import (
    DocumentDeletionRequested,
    DocumentIngestionRequested,
    DocumentReindexRequested,
)
from app.shared.enums import DocumentStatus, JobStatus
from app.shared.ids import new_uuid
from app.shared.time import to_rfc3339, utcnow
from app.storage.postgres.engine import Database
from app.storage.postgres.repositories import (
    DeletionJobRepository,
    DocumentRepository,
    IngestionJobRepository,
    ReindexJobRepository,
)

log = structlog.get_logger("recovery")


class RecoveryService:
    def __init__(self, *, database: Database, publisher, settings) -> None:  # Publisher, Settings
        self._db = database
        self._publisher = publisher
        self._settings = settings

    async def run_once(self) -> dict[str, int]:
        counts = {
            "stale_uploads": await self._cleanup_stale_uploads(),
            "ingestion_requeued": await self._requeue_ingestion(),
            "deletion_requeued": await self._requeue_deletion(),
            "reindex_requeued": await self._requeue_reindex(),
        }
        if any(counts.values()):
            log.info("recovery.ran", **counts)
        return counts

    async def _cleanup_stale_uploads(self) -> int:
        cutoff = utcnow() - timedelta(seconds=self._settings.stale_upload_ttl_seconds)
        async with self._db.begin() as session:
            docs = DocumentRepository(session)
            stale = await docs.stale_uploading(cutoff)
            for doc in stale:
                doc.status = DocumentStatus.FAILED.value
                doc.error_code = "STALE_UPLOAD"
                doc.error_message = "Upload never completed within TTL"
                doc.row_version += 1
        for _ in stale:
            metrics.recovery_actions_total.labels(kind="stale_upload").inc()
        return len(stale)

    async def _publish(self, routing_key: str, event) -> None:  # noqa: ANN001
        await self._publisher.publish(
            routing_key=routing_key,
            body=event.model_dump(mode="json"),
            message_id=event.event_id,
            trace_id=event.trace_id,
        )

    async def _requeue_ingestion(self) -> int:
        async with self._db.begin() as session:
            jobs = IngestionJobRepository(session)
            docs = DocumentRepository(session)
            stale = await jobs.find_stale()
            events = []
            for job in stale:
                doc = await docs.get(job.document_id)
                if doc is None:
                    continue
                job.status = JobStatus.QUEUED.value
                job.lease_owner = None
                job.lease_expires_at = None
                events.append(
                    DocumentIngestionRequested(
                        event_id=new_uuid(),
                        occurred_at=to_rfc3339(utcnow()),
                        domain=self._settings.domain_id,
                        document_id=job.document_id,
                        job_id=job.id,
                        attempt=job.attempt,
                        trace_id=None,
                        original_object_key=doc.original_object_key,
                        index_version=job.index_version,
                    )
                )
        for event in events:
            await self._publish(self._settings.routing_key("ingestion"), event)
            metrics.recovery_actions_total.labels(kind="ingestion_requeue").inc()
        return len(events)

    async def _requeue_deletion(self) -> int:
        async with self._db.begin() as session:
            jobs = DeletionJobRepository(session)
            stale = await jobs.find_stale()
            events = []
            for job in stale:
                job.status = JobStatus.QUEUED.value
                job.lease_owner = None
                job.lease_expires_at = None
                events.append(
                    DocumentDeletionRequested(
                        event_id=new_uuid(),
                        occurred_at=to_rfc3339(utcnow()),
                        domain=self._settings.domain_id,
                        document_id=job.document_id,
                        job_id=job.id,
                        attempt=job.attempt,
                        trace_id=None,
                        requested_by=job.requested_by,
                    )
                )
        for event in events:
            await self._publish(self._settings.routing_key("deletion"), event)
            metrics.recovery_actions_total.labels(kind="deletion_requeue").inc()
        return len(events)

    async def _requeue_reindex(self) -> int:
        async with self._db.begin() as session:
            jobs = ReindexJobRepository(session)
            stale = await jobs.find_stale()
            events = []
            for job in stale:
                job.status = JobStatus.QUEUED.value
                job.lease_owner = None
                job.lease_expires_at = None
                events.append(
                    DocumentReindexRequested(
                        event_id=new_uuid(),
                        occurred_at=to_rfc3339(utcnow()),
                        domain=self._settings.domain_id,
                        document_id=job.document_id,
                        job_id=job.id,
                        attempt=job.attempt,
                        trace_id=None,
                        source_index_version=job.source_index_version,
                        target_index_version=job.target_index_version,
                        reason=job.reason,
                    )
                )
        for event in events:
            await self._publish(self._settings.routing_key("reindex"), event)
            metrics.recovery_actions_total.labels(kind="reindex_requeue").inc()
        return len(events)
