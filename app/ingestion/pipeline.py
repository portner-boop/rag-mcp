from __future__ import annotations

import hashlib
import time

import structlog

from app.ingestion.parser.base import EmptyMarkdownError, ParserRegistry
from app.ingestion.ports import (
    EmbeddingPort,
    FinalizeData,
    IngestionStore,
    ObjectStorePort,
    VectorIndexPort,
)
from app.shared.contracts.queue import DocumentIngestionCompleted, DocumentIngestionRequested
from app.shared.enums import STAGE_PROGRESS, IngestionStage
from app.shared.errors import DomainError, ErrorCode
from app.shared.ids import new_uuid, stable_point_id
from app.shared.time import to_rfc3339, utcnow
from app.worker_support.chunk_embed import build_points, chunk_markdown, embed_texts
from app.worker_support.result import PipelineResult

log = structlog.get_logger("ingestion_pipeline")


class IngestionPipeline:
    def __init__(
        self,
        *,
        store: IngestionStore,
        object_store: ObjectStorePort,
        vector_index: VectorIndexPort,
        embedding: EmbeddingPort,
        parser_registry: ParserRegistry,
        markdown_key_for: callable,  # type: ignore[valid-type]
        chunk_size_tokens: int,
        chunk_overlap_tokens: int,
        embedding_batch_size: int,
        consumer_name: str,
        owner: str,
        lease_ttl_seconds: int,
        domain_id: str,
    ) -> None:
        self._store = store
        self._s3 = object_store
        self._vectors = vector_index
        self._embedding = embedding
        self._parsers = parser_registry
        self._markdown_key_for = markdown_key_for
        self._chunk_size = chunk_size_tokens
        self._chunk_overlap = chunk_overlap_tokens
        self._batch = embedding_batch_size
        self._consumer = consumer_name
        self._owner = owner
        self._lease_ttl = lease_ttl_seconds
        self._domain = domain_id

    async def run(self, event: DocumentIngestionRequested) -> PipelineResult:
        started = time.perf_counter()
        document_id = event.document_id
        job_id = event.job_id or ""
        self._current_stage: IngestionStage | None = None

        if await self._store.inbox_seen(self._consumer, event.event_id):
            return PipelineResult(status="duplicate")

        doc = await self._store.get_document(document_id)
        active = await self._store.get_active_index_config()
        index_version = event.index_version

        try:
            await self._store.begin_processing(
                job_id, document_id, owner=self._owner, lease_ttl_seconds=self._lease_ttl
            )

            await self._heartbeat_and_stage(job_id, IngestionStage.DOWNLOAD)
            await self._check_cancel(job_id)
            data = await self._s3.get_bytes(event.original_object_key)
            checksum = hashlib.sha256(data).hexdigest()

            if (
                doc.status == "READY"
                and doc.checksum == checksum
                and await self._vectors.count_for_document(document_id, index_version=index_version)
                > 0
            ):
                await self._store.complete_idempotent(
                    job_id=job_id, consumer=self._consumer, event_id=event.event_id
                )
                return PipelineResult(status="duplicate", chunk_count=doc.chunk_count or 0)

            await self._heartbeat_and_stage(job_id, IngestionStage.PARSING)
            await self._check_cancel(job_id)
            parser = self._parsers.get(doc.content_type)
            parsed = parser.parse(data, filename=doc.filename)
            if not parsed.markdown.strip():
                raise EmptyMarkdownError("Parsed Markdown is empty")

            await self._heartbeat_and_stage(job_id, IngestionStage.MARKDOWN_UPLOAD)
            markdown_key = self._markdown_key_for(document_id)
            await self._s3.put_bytes(
                markdown_key, parsed.markdown.encode("utf-8"), content_type="text/markdown"
            )
            await self._store.persist_markdown_key(document_id, markdown_key)

            await self._heartbeat_and_stage(job_id, IngestionStage.CHUNKING)
            await self._check_cancel(job_id)
            chunks, chunker_version = chunk_markdown(
                parsed.markdown,
                page_offsets=parsed.page_offsets,
                chunk_size_tokens=self._chunk_size,
                chunk_overlap_tokens=self._chunk_overlap,
            )
            if not chunks:
                raise EmptyMarkdownError("Document produced no chunks")
            texts = embed_texts(chunks, filename=doc.filename)

            await self._heartbeat_and_stage(job_id, IngestionStage.DENSE_EMBEDDING)
            dense = await self._embed_batched(texts, kind="dense")
            await self._heartbeat_and_stage(job_id, IngestionStage.SPARSE_EMBEDDING)
            sparse = await self._embed_batched(texts, kind="sparse")
            if len(dense) != len(chunks) or len(sparse) != len(chunks):
                raise DomainError(
                    "Embedding count does not match chunk count",
                    code=ErrorCode.INVALID_DIMENSION,
                )

            await self._heartbeat_and_stage(job_id, IngestionStage.QDRANT_UPSERT)
            await self._check_cancel(job_id)
            points = build_points(
                document_id=document_id,
                filename=doc.filename,
                content_type=doc.content_type,
                document_version=doc.document_version,
                index_version=index_version,
                chunks=chunks,
                dense=dense,
                sparse=sparse,
                point_id=lambda ci: stable_point_id(
                    document_id, doc.document_version, index_version, ci
                ),
            )
            await self._vectors.upsert(points)

            await self._heartbeat_and_stage(job_id, IngestionStage.VERIFYING)
            count = await self._vectors.count_for_document(document_id, index_version=index_version)
            if count < len(points):
                raise DomainError(
                    "Verification failed: fewer points than chunks",
                    code=ErrorCode.MISSING_POINTS,
                    retryable=True,
                    details={"expected": len(points), "actual": count},
                )
            sample_ids = [p.id for p in points[:3]]
            present = set(await self._vectors.retrieve_ids(sample_ids))
            if not set(sample_ids).issubset(present):
                raise DomainError(
                    "Verification failed: sampled points missing",
                    code=ErrorCode.MISSING_POINTS,
                    retryable=True,
                )

            await self._heartbeat_and_stage(job_id, IngestionStage.FINALIZING)
            duration_ms = int((time.perf_counter() - started) * 1000)
            completed = DocumentIngestionCompleted(
                event_id=new_uuid(),
                occurred_at=to_rfc3339(utcnow()),
                domain=self._domain,
                document_id=document_id,
                job_id=job_id,
                attempt=event.attempt,
                trace_id=event.trace_id,
                chunk_count=len(chunks),
                index_version=index_version,
                duration_ms=duration_ms,
            )
            await self._store.finalize_ready(
                document_id=document_id,
                job_id=job_id,
                consumer=self._consumer,
                event_id=event.event_id,
                data=FinalizeData(
                    chunk_count=len(chunks),
                    index_version=index_version,
                    embedding_model=active.dense_model,
                    parser_version=parsed.parser_version,
                    chunker_version=chunker_version,
                    markdown_object_key=markdown_key,
                    checksum=checksum,
                    duration_ms=duration_ms,
                    completed_event=completed.model_dump(mode="json"),
                ),
            )
            log.info(
                "ingestion.completed",
                document_id=document_id,
                job_id=job_id,
                chunks=len(chunks),
            )
            return PipelineResult(status="completed", chunk_count=len(chunks))

        except DomainError as exc:
            stage = self._current_stage.value if self._current_stage else None
            log.warning(
                "ingestion.stage_failed",
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

    async def _heartbeat_and_stage(self, job_id: str, stage: IngestionStage) -> None:
        self._current_stage = stage
        await self._store.set_stage(job_id, stage=stage.value, progress=STAGE_PROGRESS[stage])
        await self._store.heartbeat(job_id, owner=self._owner, lease_ttl_seconds=self._lease_ttl)

    async def _check_cancel(self, job_id: str) -> None:
        if await self._store.cancel_requested(job_id):
            raise DomainError("Job cancelled", code=ErrorCode.INVALID_STATE, retryable=False)

    async def _embed_batched(self, texts: list[str], *, kind: str):
        out: list = []
        for start in range(0, len(texts), self._batch):
            batch = texts[start : start + self._batch]
            if kind == "dense":
                out.extend(await self._embedding.dense(batch))
            else:
                out.extend(await self._embedding.sparse(batch))
        return out
