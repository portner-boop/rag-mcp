from __future__ import annotations

import structlog

from app.ingestion.ports import EmbeddingPort, ObjectStorePort, PointData, VectorIndexPort
from app.reindex.ports import ReindexFinalizeData, ReindexStore
from app.shared.contracts.queue import DocumentReindexCompleted, DocumentReindexRequested
from app.shared.errors import DomainError, ErrorCode
from app.shared.ids import new_uuid, stable_point_id
from app.shared.time import to_rfc3339, utcnow
from app.worker_support.chunk_embed import build_points, chunk_markdown, embed_texts
from app.worker_support.result import PipelineResult

log = structlog.get_logger("reindex_pipeline")


class ReindexPipeline:
    def __init__(
        self,
        *,
        store: ReindexStore,
        object_store: ObjectStorePort,
        vector_index: VectorIndexPort,
        embedding: EmbeddingPort,
        parser_registry,
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
        self._batch = embedding_batch_size
        self._consumer = consumer_name
        self._owner = owner
        self._lease_ttl = lease_ttl_seconds
        self._domain = domain_id

    async def run(self, event: DocumentReindexRequested) -> PipelineResult:
        document_id = event.document_id
        job_id = event.job_id or ""
        target_version = event.target_index_version
        stage = "REINDEXING"
        upserted = False

        if await self._store.inbox_seen(self._consumer, event.event_id):
            return PipelineResult(status="duplicate")

        try:
            doc = await self._store.get_document(document_id)
            target_cfg = await self._store.get_index_config(target_version)
            await self._store.begin_processing(
                job_id, document_id, owner=self._owner, lease_ttl_seconds=self._lease_ttl
            )

            stage = "READ_SOURCE"
            await self._store.set_stage(job_id, stage=stage, progress=20)
            await self._heartbeat(job_id)
            await self._check_cancel(job_id)
            markdown, parser_version, page_offsets = await self._read_markdown(doc)

            stage = "CHUNKING"
            await self._store.set_stage(job_id, stage=stage, progress=40)
            chunks, chunker_version = chunk_markdown(
                markdown,
                page_offsets=page_offsets,
                chunk_size_tokens=target_cfg.chunk_size_tokens,
                chunk_overlap_tokens=target_cfg.chunk_overlap_tokens,
            )
            if not chunks:
                raise DomainError("Reindex produced no chunks", code=ErrorCode.CORRUPTED_FILE)

            stage = "EMBEDDING"
            await self._store.set_stage(job_id, stage=stage, progress=65)
            await self._check_cancel(job_id)
            texts = embed_texts(chunks, filename=doc.filename)
            dense = await self._embed(texts, kind="dense")
            sparse = await self._embed(texts, kind="sparse")

            stage = "QDRANT_UPSERT"
            await self._store.set_stage(job_id, stage=stage, progress=80)
            points = build_points(
                document_id=document_id,
                filename=doc.filename,
                content_type=doc.content_type,
                document_version=doc.document_version,
                index_version=target_version,
                chunks=chunks,
                dense=dense,
                sparse=sparse,
                point_id=lambda ci: stable_point_id(
                    document_id, doc.document_version, target_version, ci
                ),
            )
            await self._vectors.upsert(points)
            upserted = True

            stage = "VERIFYING"
            await self._store.set_stage(job_id, stage=stage, progress=90)
            count = await self._vectors.count_for_document(
                document_id, index_version=target_version
            )
            if count < len(points):
                raise DomainError(
                    "Reindex verification failed: fewer points than chunks",
                    code=ErrorCode.MISSING_POINTS,
                    retryable=True,
                    details={"expected": len(points), "actual": count},
                )

            stage = "CUTOVER"
            await self._store.set_stage(job_id, stage=stage, progress=97)
            completed = DocumentReindexCompleted(
                event_id=new_uuid(),
                occurred_at=to_rfc3339(utcnow()),
                domain=self._domain,
                document_id=document_id,
                job_id=job_id,
                attempt=event.attempt,
                trace_id=event.trace_id,
                target_index_version=target_version,
                chunk_count=len(chunks),
            )
            await self._store.finalize_cutover(
                document_id=document_id,
                job_id=job_id,
                consumer=self._consumer,
                event_id=event.event_id,
                data=ReindexFinalizeData(
                    target_index_version=target_version,
                    source_index_version=event.source_index_version,
                    chunk_count=len(chunks),
                    embedding_model=target_cfg.dense_model,
                    parser_version=parser_version,
                    chunker_version=chunker_version,
                    completed_event=completed.model_dump(mode="json"),
                ),
            )

            if (
                event.source_index_version is not None
                and event.source_index_version != target_version
            ):
                try:
                    await self._vectors.delete_document(
                        document_id, index_version=event.source_index_version
                    )
                except DomainError:
                    log.warning("reindex.old_cleanup_deferred", document_id=document_id)

            log.info(
                "reindex.completed",
                document_id=document_id,
                target_index_version=target_version,
                chunks=len(chunks),
            )
            return PipelineResult(status="completed", chunk_count=len(chunks))

        except DomainError as exc:
            if upserted:
                try:
                    await self._vectors.delete_document(document_id, index_version=target_version)
                except DomainError:
                    pass
            log.warning(
                "reindex.stage_failed",
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

    async def _read_markdown(self, doc) -> tuple[str, str, list[int] | None]:  # noqa: ANN001
        if doc.markdown_object_key and await self._s3.exists(doc.markdown_object_key):
            data = await self._s3.get_bytes(doc.markdown_object_key)
            return data.decode("utf-8"), "reused-markdown", None
        data = await self._s3.get_bytes(doc.original_object_key)
        parsed = self._parsers.get(doc.content_type).parse(data, filename=doc.filename)
        if not parsed.markdown.strip():
            raise DomainError("Parsed Markdown is empty", code=ErrorCode.CORRUPTED_FILE)
        return parsed.markdown, parsed.parser_version, parsed.page_offsets

    async def _embed(self, texts: list[str], *, kind: str):
        out: list = []
        for start in range(0, len(texts), self._batch):
            batch = texts[start : start + self._batch]
            if kind == "dense":
                out.extend(await self._embedding.dense(batch))
            else:
                out.extend(await self._embedding.sparse(batch))
        return out

    async def _check_cancel(self, job_id: str) -> None:
        if await self._store.cancel_requested(job_id):
            raise DomainError("Reindex cancelled", code=ErrorCode.INVALID_STATE, retryable=False)

    async def _heartbeat(self, job_id: str) -> None:
        await self._store.heartbeat(job_id, owner=self._owner, lease_ttl_seconds=self._lease_ttl)


__all__ = ["ReindexPipeline", "PipelineResult", "PointData"]
