"""Compensation: idempotent delete-on-missing and reindex temp-point cleanup (spec 11, 12.3)."""

from __future__ import annotations

from app.deletion.pipeline import DeletionPipeline
from app.ingestion.ports import DocumentState, IndexConfigState
from app.reindex.pipeline import ReindexPipeline
from app.shared.contracts.queue import DocumentDeletionRequested, DocumentReindexRequested
from app.shared.enums import DocumentStatus
from app.shared.ids import new_uuid
from app.shared.time import to_rfc3339, utcnow
from app.testing.fakes import (
    FakeDeletionStore,
    FakeEmbedding,
    FakeObjectStore,
    FakeReindexStore,
    FakeVectorIndex,
)

DIM = 64


def _doc(status: DocumentStatus, document_id: str, *, index_version=1) -> DocumentState:
    return DocumentState(
        document_id=document_id,
        filename="d.md",
        content_type="text/markdown",
        size=10,
        status=status.value,
        document_version=1,
        checksum="x",
        original_object_key=f"documents/{document_id}/original/d.md",
        markdown_object_key=f"documents/{document_id}/parsed/document.md",
        index_version=index_version,
    )


async def test_delete_on_already_missing_artifacts_succeeds() -> None:
    document_id = new_uuid()
    store = FakeDeletionStore(doc=_doc(DocumentStatus.DELETING, document_id))
    pipeline = DeletionPipeline(
        store=store,
        object_store=FakeObjectStore(),
        vector_index=FakeVectorIndex(),
        consumer_name="deletion-worker",
        owner="w",
        lease_ttl_seconds=120,
        domain_id="hr",
    )
    event = DocumentDeletionRequested(
        event_id=new_uuid(),
        occurred_at=to_rfc3339(utcnow()),
        domain="hr",
        document_id=document_id,
        job_id="j",
        attempt=0,
        requested_by="admin",
    )
    result = await pipeline.run(event)
    assert result.status == "completed"  # missing points/objects treated as success
    assert store.doc.status == DocumentStatus.DELETED.value


async def test_reindex_verify_failure_cleans_temp_target_points() -> None:
    document_id = new_uuid()
    cfg = IndexConfigState(
        version=1,
        dense_model="m",
        dense_dimension=DIM,
        sparse_model="s",
        qdrant_collection="c",
        chunk_size_tokens=40,
        chunk_overlap_tokens=8,
    )
    tgt = IndexConfigState(
        version=2,
        dense_model="m",
        dense_dimension=DIM,
        sparse_model="s",
        qdrant_collection="c",
        chunk_size_tokens=40,
        chunk_overlap_tokens=8,
    )
    store = FakeReindexStore(
        doc=_doc(DocumentStatus.REINDEXING, document_id), configs={1: cfg, 2: tgt}, active_version=1
    )
    s3 = FakeObjectStore()
    s3.objects[store.doc.markdown_object_key] = (
        b"# H\n\n" + b"policy words here " * 30,
        "text/markdown",
    )
    vectors = FakeVectorIndex()

    # Force verification to see zero points at the target version after upsert.
    async def _count_zero(document_id, *, index_version):
        return 0

    vectors.count_for_document = _count_zero  # type: ignore[method-assign]

    pipeline = ReindexPipeline(
        store=store,
        object_store=s3,
        vector_index=vectors,
        embedding=FakeEmbedding(DIM),
        parser_registry=None,
        embedding_batch_size=16,
        consumer_name="reindex-worker",
        owner="w",
        lease_ttl_seconds=120,
        domain_id="hr",
    )
    event = DocumentReindexRequested(
        event_id=new_uuid(),
        occurred_at=to_rfc3339(utcnow()),
        domain="hr",
        document_id=document_id,
        job_id="j",
        attempt=0,
        source_index_version=1,
        target_index_version=2,
        reason=None,
    )
    result = await pipeline.run(event)
    assert result.status == "failed" and result.retryable
    assert store.active_version == 1  # never cut over
    # Temporary target points were compensated away.
    assert all(p.payload.get("index_version") != 2 for p in vectors.points.values())
