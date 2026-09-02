from __future__ import annotations

from datetime import timedelta

from app.config import Settings
from app.shared.contracts.mcp import (
    GetIngestionStatusInput,
    IngestionStatusOutput,
    PrepareUploadInput,
    PrepareUploadOutput,
    StartIngestionInput,
    StartIngestionOutput,
)
from app.shared.contracts.queue import DocumentIngestionRequested
from app.shared.enums import DocumentStatus, EventType, JobStatus
from app.shared.errors import IdempotencyConflictError, ValidationError
from app.shared.ids import new_uuid
from app.shared.time import to_rfc3339, utcnow
from app.shared.trace import current_trace_id
from app.storage.keys import original_key
from app.storage.postgres.engine import Database
from app.storage.postgres.repositories import (
    DocumentRepository,
    EventRepository,
    IndexConfigRepository,
    IngestionJobRepository,
    OutboxRepository,
)
from app.storage.s3 import S3ObjectStore


class IngestionService:
    def __init__(
        self, *, database: Database, object_store: S3ObjectStore, settings: Settings
    ) -> None:
        self._db = database
        self._s3 = object_store
        self._settings = settings

    async def prepare_document_upload(self, payload: PrepareUploadInput) -> PrepareUploadOutput:
        s = self._settings
        if payload.size > s.upload_max_bytes:
            raise ValidationError(
                "Upload exceeds maximum size",
                details={"max_bytes": s.upload_max_bytes},
            )
        if payload.content_type not in s.allowed_content_types:
            raise ValidationError(
                "Unsupported content type",
                details={"content_type": payload.content_type},
            )

        document_id = new_uuid()
        key = original_key(document_id, payload.filename)
        async with self._db.session() as session:
            repo = DocumentRepository(session)
            doc = await repo.create_uploading(
                filename=payload.filename,
                content_type=payload.content_type,
                size=payload.size,
                checksum=payload.checksum,
                original_object_key=key,
                created_by=payload.created_by,
            )
            document_id = doc.id

        upload_url = await self._s3.presign_put(
            key, content_type=payload.content_type, expires_in=s.presigned_upload_ttl_seconds
        )
        expires_at = utcnow() + timedelta(seconds=s.presigned_upload_ttl_seconds)
        return PrepareUploadOutput(
            document_id=document_id,
            upload_url=upload_url,
            upload_headers={"Content-Type": payload.content_type},
            expires_at=to_rfc3339(expires_at),
        )

    async def start_document_ingestion(self, payload: StartIngestionInput) -> StartIngestionOutput:
        async with self._db.session() as session:
            docs = DocumentRepository(session)
            jobs = IngestionJobRepository(session)
            outbox = OutboxRepository(session)
            events = EventRepository(session)
            index_configs = IndexConfigRepository(session)

            existing = await jobs.get_by_idempotency_key(payload.idempotency_key)
            if existing is not None:
                if existing.document_id != payload.document_id:
                    raise IdempotencyConflictError(
                        "Idempotency key reused with a different document",
                        details={"idempotency_key": payload.idempotency_key},
                    )
                in_flight = {
                    JobStatus.QUEUED.value,
                    JobStatus.PROCESSING.value,
                    JobStatus.RETRY_WAIT.value,
                }
                status = JobStatus.QUEUED.value if existing.status in in_flight else existing.status
                return StartIngestionOutput(
                    document_id=existing.document_id, job_id=existing.id, status=status
                )

            doc = await docs.get_for_update(payload.document_id)

            meta = await self._s3.head(doc.original_object_key)
            if meta.size != doc.size:
                raise ValidationError(
                    "Uploaded object size does not match declared size",
                    details={"expected": doc.size, "actual": meta.size},
                )
            if payload.checksum is not None:
                if doc.checksum is not None and doc.checksum != payload.checksum:
                    raise IdempotencyConflictError("Checksum does not match the document record")
                doc.checksum = payload.checksum

            active = await index_configs.get_active_or_raise()

            if DocumentStatus(doc.status) is DocumentStatus.UPLOADING:
                await docs.transition(
                    doc, allowed_from={DocumentStatus.UPLOADING}, to=DocumentStatus.UPLOADED
                )
            await docs.transition(
                doc,
                allowed_from={DocumentStatus.UPLOADED, DocumentStatus.FAILED},
                to=DocumentStatus.QUEUED,
            )

            job = await jobs.create(
                document_id=doc.id,
                idempotency_key=payload.idempotency_key,
                index_version=active.version,
                max_attempts=self._settings.max_attempts,
            )

            trace_id = current_trace_id()
            event = DocumentIngestionRequested(
                event_id=new_uuid(),
                occurred_at=to_rfc3339(utcnow()),
                domain=self._settings.domain_id,
                document_id=doc.id,
                job_id=job.id,
                attempt=0,
                trace_id=trace_id,
                original_object_key=doc.original_object_key,
                index_version=active.version,
            )
            await outbox.add(
                event_id=event.event_id,
                event_type=EventType.DOCUMENT_INGESTION_REQUESTED.value,
                aggregate_id=doc.id,
                payload=event.model_dump(mode="json"),
                routing_key=self._settings.routing_key("ingestion"),
            )
            await events.append(
                document_id=doc.id,
                event_type=EventType.DOCUMENT_INGESTION_REQUESTED.value,
                payload={"job_id": job.id, "index_version": active.version},
                trace_id=trace_id,
            )

            return StartIngestionOutput(
                document_id=doc.id, job_id=job.id, status=JobStatus.QUEUED.value
            )

    async def get_ingestion_status(self, payload: GetIngestionStatusInput) -> IngestionStatusOutput:
        async with self._db.session() as session:
            jobs = IngestionJobRepository(session)
            job = await jobs.get_or_raise(payload.job_id)
            error = None
            if job.error_code:
                error = {"code": job.error_code, "message": job.error_message}
            return IngestionStatusOutput(
                job_id=job.id,
                document_id=job.document_id,
                status=job.status,
                stage=job.stage,
                progress=job.progress,
                attempt=job.attempt,
                error=error,
            )
