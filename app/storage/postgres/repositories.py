from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import and_, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.shared.enums import SEARCH_EXCLUDED_STATUSES, DocumentStatus, JobStatus, OutboxStatus
from app.shared.errors import InvalidStateError, NotFoundError
from app.shared.ids import new_uuid
from app.shared.time import utcnow
from app.storage.postgres.models import (
    DeletionJob,
    Document,
    DocumentEvent,
    InboxEvent,
    IndexConfig,
    IngestionJob,
    OutboxEvent,
    ReindexJob,
)


class DocumentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create_uploading(
        self,
        *,
        filename: str,
        content_type: str,
        size: int,
        checksum: str | None,
        original_object_key: str,
        created_by: str,
    ) -> Document:
        doc = Document(
            id=new_uuid(),
            filename=filename,
            content_type=content_type,
            size=size,
            checksum=checksum,
            original_object_key=original_object_key,
            status=DocumentStatus.UPLOADING.value,
            document_version=1,
            created_by=created_by,
        )
        self.session.add(doc)
        await self.session.flush()
        return doc

    async def get(self, document_id: str) -> Document | None:
        return await self.session.get(Document, document_id)

    async def get_or_raise(self, document_id: str) -> Document:
        doc = await self.get(document_id)
        if doc is None:
            raise NotFoundError("Document not found", details={"document_id": document_id})
        return doc

    async def get_for_update(self, document_id: str) -> Document:
        stmt = select(Document).where(Document.id == document_id).with_for_update()
        doc = (await self.session.execute(stmt)).scalar_one_or_none()
        if doc is None:
            raise NotFoundError("Document not found", details={"document_id": document_id})
        return doc

    async def transition(
        self, doc: Document, *, allowed_from: set[DocumentStatus], to: DocumentStatus
    ) -> None:
        current = DocumentStatus(doc.status)
        if current not in allowed_from:
            raise InvalidStateError(
                f"Illegal transition {current.value} -> {to.value}",
                details={"from": current.value, "to": to.value},
            )
        doc.status = to.value
        doc.row_version += 1

    async def find(
        self,
        *,
        status: DocumentStatus | None,
        query: str | None,
        filename: str | None,
        limit: int,
    ) -> list[Document]:
        stmt = select(Document)
        if status is not None:
            stmt = stmt.where(Document.status == status.value)
        if filename:
            stmt = stmt.where(func.lower(Document.filename).like(f"%{filename.lower()}%"))
        if query:
            stmt = stmt.where(func.lower(Document.filename).like(f"%{query.lower()}%"))
        stmt = stmt.order_by(Document.created_at.desc()).limit(limit)
        return list((await self.session.execute(stmt)).scalars().all())

    async def stale_uploading(self, older_than: datetime) -> list[Document]:
        stmt = select(Document).where(
            and_(
                Document.status == DocumentStatus.UPLOADING.value,
                Document.created_at < older_than,
            )
        )
        return list((await self.session.execute(stmt)).scalars().all())

    async def find_indexed_by_checksum(
        self, checksum: str, document_version: int
    ) -> Document | None:
        stmt = select(Document).where(
            and_(
                Document.checksum == checksum,
                Document.document_version == document_version,
                Document.status == DocumentStatus.READY.value,
            )
        )
        return (await self.session.execute(stmt)).scalars().first()

    async def excluded_ids(self, *, limit: int = 10000) -> list[str]:
        stmt = (
            select(Document.id)
            .where(Document.status.in_([s.value for s in SEARCH_EXCLUDED_STATUSES]))
            .limit(limit)
        )
        return list((await self.session.execute(stmt)).scalars().all())


class IngestionJobRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_by_idempotency_key(self, key: str) -> IngestionJob | None:
        stmt = select(IngestionJob).where(IngestionJob.idempotency_key == key)
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def create(
        self, *, document_id: str, idempotency_key: str, index_version: int, max_attempts: int
    ) -> IngestionJob:
        job = IngestionJob(
            id=new_uuid(),
            document_id=document_id,
            status=JobStatus.QUEUED.value,
            idempotency_key=idempotency_key,
            index_version=index_version,
            max_attempts=max_attempts,
            available_at=utcnow(),
        )
        self.session.add(job)
        await self.session.flush()
        return job

    async def get(self, job_id: str) -> IngestionJob | None:
        return await self.session.get(IngestionJob, job_id)

    async def get_or_raise(self, job_id: str) -> IngestionJob:
        job = await self.get(job_id)
        if job is None:
            raise NotFoundError("Job not found", details={"job_id": job_id})
        return job

    async def get_for_update(self, job_id: str) -> IngestionJob:
        stmt = select(IngestionJob).where(IngestionJob.id == job_id).with_for_update()
        job = (await self.session.execute(stmt)).scalar_one_or_none()
        if job is None:
            raise NotFoundError("Job not found", details={"job_id": job_id})
        return job

    async def acquire_lease(
        self, job_id: str, *, owner: str, lease_ttl_seconds: int
    ) -> IngestionJob | None:
        now = utcnow()
        job = await self.get_for_update(job_id)
        if job.status in (JobStatus.COMPLETED.value, JobStatus.CANCELLED.value):
            return job
        lease_live = job.lease_expires_at is not None and job.lease_expires_at > now
        if lease_live and job.lease_owner != owner:
            return None
        job.lease_owner = owner
        job.lease_expires_at = now + timedelta(seconds=lease_ttl_seconds)
        job.heartbeat_at = now
        return job

    async def heartbeat(self, job_id: str, *, owner: str, lease_ttl_seconds: int) -> bool:
        now = utcnow()
        stmt = (
            update(IngestionJob)
            .where(and_(IngestionJob.id == job_id, IngestionJob.lease_owner == owner))
            .values(
                heartbeat_at=now,
                lease_expires_at=now + timedelta(seconds=lease_ttl_seconds),
            )
        )
        result = await self.session.execute(stmt)
        return result.rowcount > 0

    async def set_progress(
        self, job: IngestionJob, *, status: JobStatus, stage: str | None, progress: int
    ) -> None:
        job.status = status.value
        job.stage = stage
        job.progress = progress

    async def find_stale(self, *, limit: int = 100) -> list[IngestionJob]:
        return await _find_stale(self.session, IngestionJob, limit=limit)


async def _find_stale(session: AsyncSession, model, *, limit: int):  # noqa: ANN001
    now = utcnow()
    stmt = (
        select(model)
        .where(
            and_(
                model.status == JobStatus.PROCESSING.value,
                model.lease_expires_at.is_not(None),
                model.lease_expires_at < now,
            )
        )
        .limit(limit)
    )
    return list((await session.execute(stmt)).scalars().all())


class _JobRepositoryBase:
    _model: type

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_by_idempotency_key(self, key: str):
        stmt = select(self._model).where(self._model.idempotency_key == key)
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def get(self, job_id: str):
        return await self.session.get(self._model, job_id)

    async def get_or_raise(self, job_id: str):
        job = await self.get(job_id)
        if job is None:
            raise NotFoundError("Job not found", details={"job_id": job_id})
        return job

    async def get_for_update(self, job_id: str):
        stmt = select(self._model).where(self._model.id == job_id).with_for_update()
        job = (await self.session.execute(stmt)).scalar_one_or_none()
        if job is None:
            raise NotFoundError("Job not found", details={"job_id": job_id})
        return job

    async def acquire_lease(self, job_id: str, *, owner: str, lease_ttl_seconds: int):
        now = utcnow()
        job = await self.get_for_update(job_id)
        if job.status in (JobStatus.COMPLETED.value, JobStatus.CANCELLED.value):
            return job
        lease_live = job.lease_expires_at is not None and job.lease_expires_at > now
        if lease_live and job.lease_owner != owner:
            return None
        job.lease_owner = owner
        job.lease_expires_at = now + timedelta(seconds=lease_ttl_seconds)
        job.heartbeat_at = now
        return job

    async def heartbeat(self, job_id: str, *, owner: str, lease_ttl_seconds: int) -> bool:
        now = utcnow()
        stmt = (
            update(self._model)
            .where(and_(self._model.id == job_id, self._model.lease_owner == owner))
            .values(heartbeat_at=now, lease_expires_at=now + timedelta(seconds=lease_ttl_seconds))
        )
        result = await self.session.execute(stmt)
        return result.rowcount > 0

    async def find_stale(self, *, limit: int = 100):
        return await _find_stale(self.session, self._model, limit=limit)


class DeletionJobRepository(_JobRepositoryBase):
    _model = DeletionJob

    async def create(
        self, *, document_id: str, idempotency_key: str, requested_by: str, max_attempts: int
    ) -> DeletionJob:
        job = DeletionJob(
            id=new_uuid(),
            document_id=document_id,
            status=JobStatus.QUEUED.value,
            idempotency_key=idempotency_key,
            requested_by=requested_by,
            max_attempts=max_attempts,
            available_at=utcnow(),
        )
        self.session.add(job)
        await self.session.flush()
        return job

    async def find_active_for_document(self, document_id: str) -> DeletionJob | None:
        stmt = (
            select(DeletionJob)
            .where(
                and_(
                    DeletionJob.document_id == document_id,
                    DeletionJob.status.in_(
                        [
                            JobStatus.QUEUED.value,
                            JobStatus.PROCESSING.value,
                            JobStatus.RETRY_WAIT.value,
                        ]
                    ),
                )
            )
            .order_by(DeletionJob.created_at.desc())
        )
        return (await self.session.execute(stmt)).scalars().first()


class ReindexJobRepository(_JobRepositoryBase):
    _model = ReindexJob

    async def create(
        self,
        *,
        document_id: str,
        idempotency_key: str,
        source_index_version: int | None,
        target_index_version: int,
        reason: str | None,
        max_attempts: int,
    ) -> ReindexJob:
        job = ReindexJob(
            id=new_uuid(),
            document_id=document_id,
            status=JobStatus.QUEUED.value,
            idempotency_key=idempotency_key,
            source_index_version=source_index_version,
            target_index_version=target_index_version,
            reason=reason,
            max_attempts=max_attempts,
            available_at=utcnow(),
        )
        self.session.add(job)
        await self.session.flush()
        return job

    async def get_active_target(
        self, document_id: str, target_index_version: int
    ) -> ReindexJob | None:
        stmt = select(ReindexJob).where(
            and_(
                ReindexJob.document_id == document_id,
                ReindexJob.target_index_version == target_index_version,
            )
        )
        return (await self.session.execute(stmt)).scalar_one_or_none()


class OutboxRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add(
        self,
        *,
        event_id: str,
        event_type: str,
        aggregate_id: str,
        payload: dict[str, Any],
        routing_key: str,
    ) -> OutboxEvent:
        row = OutboxEvent(
            event_id=event_id,
            event_type=event_type,
            aggregate_id=aggregate_id,
            payload=payload,
            routing_key=routing_key,
            status=OutboxStatus.PENDING.value,
            available_at=utcnow(),
        )
        self.session.add(row)
        await self.session.flush()
        return row

    async def fetch_pending(self, limit: int = 50) -> list[OutboxEvent]:
        stmt = (
            select(OutboxEvent)
            .where(
                and_(
                    OutboxEvent.status == OutboxStatus.PENDING.value,
                    OutboxEvent.available_at <= utcnow(),
                )
            )
            .order_by(OutboxEvent.available_at)
            .limit(limit)
            .with_for_update(skip_locked=True)
        )
        return list((await self.session.execute(stmt)).scalars().all())

    async def mark_published(self, row: OutboxEvent) -> None:
        row.status = OutboxStatus.PUBLISHED.value
        row.published_at = utcnow()

    async def mark_failed(self, row: OutboxEvent, *, backoff_seconds: int) -> None:
        row.attempt += 1
        row.available_at = utcnow() + timedelta(seconds=backoff_seconds)


class InboxRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def seen(self, consumer: str, event_id: str) -> bool:
        row = await self.session.get(InboxEvent, {"consumer": consumer, "event_id": event_id})
        return row is not None

    async def record(self, consumer: str, event_id: str, result_hash: str | None = None) -> None:
        self.session.add(InboxEvent(consumer=consumer, event_id=event_id, result_hash=result_hash))
        await self.session.flush()


class EventRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def append(
        self,
        *,
        document_id: str,
        event_type: str,
        payload: dict[str, Any],
        trace_id: str | None,
    ) -> None:
        self.session.add(
            DocumentEvent(
                id=new_uuid(),
                document_id=document_id,
                event_type=event_type,
                payload=payload,
                trace_id=trace_id,
            )
        )
        await self.session.flush()


class IndexConfigRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_active(self) -> IndexConfig | None:
        stmt = select(IndexConfig).where(IndexConfig.active.is_(True))
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def get_active_or_raise(self) -> IndexConfig:
        cfg = await self.get_active()
        if cfg is None:
            raise InvalidStateError("No active index config")
        return cfg

    async def get_by_version(self, version: int) -> IndexConfig | None:
        stmt = select(IndexConfig).where(IndexConfig.version == version)
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def create(
        self,
        *,
        version: int,
        dense_model: str,
        dense_dimension: int,
        sparse_model: str | None,
        reranker_model: str | None,
        chunk_size_tokens: int,
        chunk_overlap_tokens: int,
        qdrant_collection: str,
        active: bool = False,
    ) -> IndexConfig:
        cfg = IndexConfig(
            id=new_uuid(),
            version=version,
            dense_model=dense_model,
            dense_dimension=dense_dimension,
            sparse_model=sparse_model,
            reranker_model=reranker_model,
            chunk_size_tokens=chunk_size_tokens,
            chunk_overlap_tokens=chunk_overlap_tokens,
            qdrant_collection=qdrant_collection,
            active=active,
        )
        self.session.add(cfg)
        await self.session.flush()
        return cfg

    async def activate(self, version: int) -> None:
        await self.session.execute(update(IndexConfig).values(active=False))
        await self.session.execute(
            update(IndexConfig).where(IndexConfig.version == version).values(active=True)
        )


__all__ = [
    "DocumentRepository",
    "IngestionJobRepository",
    "DeletionJobRepository",
    "ReindexJobRepository",
    "OutboxRepository",
    "InboxRepository",
    "EventRepository",
    "IndexConfigRepository",
]
