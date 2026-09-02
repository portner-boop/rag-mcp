"""Smoke check: deletion + reindex lifecycle on in-memory fakes (D04 checklist).

Proves: delete removes points/objects and marks DELETED; repeated delete is a no-op;
reindex builds the target version, verifies, then cuts over (activates target) and cleans
old-version points; a failed reindex leaves the old version active; cancellation stops
reindex before cutover.

Run: uv run python scripts/smoke_lifecycle.py
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from app.deletion.pipeline import DeletionPipeline
from app.ingestion.ports import DocumentState, IndexConfigState, PointData
from app.reindex.pipeline import ReindexPipeline
from app.shared.contracts.queue import DocumentDeletionRequested, DocumentReindexRequested
from app.shared.enums import DocumentStatus
from app.shared.errors import DomainError, ErrorCode
from app.shared.ids import new_uuid
from app.shared.time import to_rfc3339, utcnow
from app.testing.embeddings import dense_vector, sparse_vector
from app.testing.fakes import FakeEmbedding, FakeObjectStore, FakeVectorIndex

DIMENSION = 64
DOMAIN = "hr"


# --- compact fake stores (smoke-only) --------------------------------------------------
@dataclass
class _Job:
    status: str = "QUEUED"
    stage: str | None = None
    cancel: bool = False


@dataclass
class FakeDeletionStore:
    doc: DocumentState
    job: _Job = field(default_factory=_Job)
    inbox: set = field(default_factory=set)
    outbox: list = field(default_factory=list)

    async def inbox_seen(self, consumer, event_id):
        return (consumer, event_id) in self.inbox

    async def get_document(self, document_id):
        return self.doc

    async def begin_processing(self, job_id, document_id, *, owner, lease_ttl_seconds):
        self.doc.status = DocumentStatus.DELETING.value
        self.job.status = "PROCESSING"

    async def set_stage(self, job_id, *, stage, progress):
        self.job.stage = stage

    async def heartbeat(self, job_id, *, owner, lease_ttl_seconds):
        return True

    async def finalize_deleted(self, *, document_id, job_id, consumer, event_id, deleted_event):
        self.doc.status = DocumentStatus.DELETED.value
        self.job.status = "COMPLETED"
        self.outbox.append(deleted_event)
        self.inbox.add((consumer, event_id))

    async def mark_failed(
        self, *, document_id, job_id, set_document_failed, consumer, event_id, **_
    ):
        if set_document_failed:
            self.doc.status = DocumentStatus.DELETE_FAILED.value
            self.inbox.add((consumer, event_id))

    async def record_retry(self, job_id, *, attempt, available_in):
        self.job.status = "RETRY_WAIT"


@dataclass
class FakeReindexStore:
    doc: DocumentState
    configs: dict  # version -> IndexConfigState
    active_version: int
    job: _Job = field(default_factory=_Job)
    inbox: set = field(default_factory=set)
    outbox: list = field(default_factory=list)

    async def inbox_seen(self, consumer, event_id):
        return (consumer, event_id) in self.inbox

    async def get_document(self, document_id):
        return self.doc

    async def get_index_config(self, version):
        return self.configs[version]

    async def begin_processing(self, job_id, document_id, *, owner, lease_ttl_seconds):
        self.job.status = "PROCESSING"

    async def set_stage(self, job_id, *, stage, progress):
        self.job.stage = stage

    async def heartbeat(self, job_id, *, owner, lease_ttl_seconds):
        return True

    async def cancel_requested(self, job_id):
        return self.job.cancel

    async def finalize_cutover(self, *, document_id, job_id, consumer, event_id, data):
        self.doc.status = DocumentStatus.READY.value
        self.doc.index_version = data.target_index_version
        self.active_version = data.target_index_version  # cutover
        self.job.status = "COMPLETED"
        self.outbox.append(data.completed_event)
        self.inbox.add((consumer, event_id))

    async def mark_failed(
        self, *, document_id, job_id, set_document_failed, consumer, event_id, **_
    ):
        if set_document_failed:
            self.doc.status = DocumentStatus.READY.value  # restore old; active unchanged
            self.inbox.add((consumer, event_id))

    async def record_retry(self, job_id, *, attempt, available_in):
        self.job.status = "RETRY_WAIT"


def _seed_points(vectors: FakeVectorIndex, document_id: str, *, index_version: int, n: int = 3):
    for ci in range(n):
        text = f"chunk {ci} about leave transfer policy details"
        sv = sparse_vector(text)
        pid = f"{document_id}:{ci}:{index_version}"
        vectors.points[pid] = PointData(
            id=pid,
            dense=dense_vector(text, DIMENSION),
            sparse_indices=sv.indices,
            sparse_values=sv.values,
            payload={
                "document_id": document_id,
                "index_version": index_version,
                "document_version": 1,
                "text": text,
                "filename": "d.md",
            },
        )


async def run_deletion() -> None:
    print("deletion lifecycle smoke:")
    document_id, job_id = new_uuid(), new_uuid()
    okey, mkey = (
        f"documents/{document_id}/original/d.md",
        f"documents/{document_id}/parsed/document.md",
    )
    doc = DocumentState(
        document_id=document_id,
        filename="d.md",
        content_type="text/markdown",
        size=10,
        status=DocumentStatus.DELETING.value,
        document_version=1,
        checksum="x",
        original_object_key=okey,
        markdown_object_key=mkey,
        index_version=1,
    )
    store = FakeDeletionStore(doc=doc)
    s3 = FakeObjectStore()
    s3.objects[okey] = (b"orig", "text/markdown")
    s3.objects[mkey] = (b"# md", "text/markdown")
    vectors = FakeVectorIndex()
    _seed_points(vectors, document_id, index_version=1)

    def pipeline():
        return DeletionPipeline(
            store=store,
            object_store=s3,
            vector_index=vectors,
            consumer_name="deletion-worker",
            owner="w1",
            lease_ttl_seconds=120,
            domain_id=DOMAIN,
        )

    event = DocumentDeletionRequested(
        event_id=new_uuid(),
        occurred_at=to_rfc3339(utcnow()),
        domain=DOMAIN,
        document_id=document_id,
        job_id=job_id,
        attempt=0,
        requested_by="admin",
    )
    r = await pipeline().run(event)
    assert r.status == "completed", r
    assert doc.status == DocumentStatus.DELETED.value
    assert await vectors.count_all_for_document(document_id) == 0
    assert okey not in s3.objects and mkey not in s3.objects
    assert any(e["event_type"] == "DocumentDeleted" for e in store.outbox)
    print("  OK  delete removed points+objects and marked DELETED")

    r2 = await pipeline().run(event)
    assert r2.status == "duplicate", r2
    print("  OK  repeated delete is an idempotent no-op")


async def run_reindex() -> None:
    print("reindex lifecycle smoke:")
    document_id, job_id = new_uuid(), new_uuid()
    okey = f"documents/{document_id}/original/d.md"
    mkey = f"documents/{document_id}/parsed/document.md"
    doc = DocumentState(
        document_id=document_id,
        filename="d.md",
        content_type="text/markdown",
        size=10,
        status=DocumentStatus.REINDEXING.value,
        document_version=1,
        checksum="x",
        original_object_key=okey,
        markdown_object_key=mkey,
        index_version=1,
    )
    cfg1 = IndexConfigState(
        version=1,
        dense_model="m",
        dense_dimension=DIMENSION,
        sparse_model="s",
        qdrant_collection="hr-knowledge",
        chunk_size_tokens=40,
        chunk_overlap_tokens=8,
    )
    cfg2 = IndexConfigState(
        version=2,
        dense_model="m",
        dense_dimension=DIMENSION,
        sparse_model="s",
        qdrant_collection="hr-knowledge",
        chunk_size_tokens=40,
        chunk_overlap_tokens=8,
    )
    store = FakeReindexStore(doc=doc, configs={1: cfg1, 2: cfg2}, active_version=1)
    s3 = FakeObjectStore()
    s3.objects[mkey] = (
        b"# HR\n\n" + b"leave transfer policy details apply here " * 20,
        "text/markdown",
    )
    vectors = FakeVectorIndex()
    _seed_points(vectors, document_id, index_version=1)  # old active points
    embedding = FakeEmbedding(DIMENSION)

    def pipeline():
        return ReindexPipeline(
            store=store,
            object_store=s3,
            vector_index=vectors,
            embedding=embedding,
            parser_registry=None,
            embedding_batch_size=16,
            consumer_name="reindex-worker",
            owner="w1",
            lease_ttl_seconds=120,
            domain_id=DOMAIN,
        )

    event = DocumentReindexRequested(
        event_id=new_uuid(),
        occurred_at=to_rfc3339(utcnow()),
        domain=DOMAIN,
        document_id=document_id,
        job_id=job_id,
        attempt=0,
        source_index_version=1,
        target_index_version=2,
        reason="new chunker",
    )
    r = await pipeline().run(event)
    assert r.status == "completed", r
    assert store.active_version == 2, "cutover did not switch active version"
    assert doc.index_version == 2 and doc.status == DocumentStatus.READY.value
    assert await vectors.count_for_document(document_id, index_version=2) == r.chunk_count
    assert await vectors.count_for_document(document_id, index_version=1) == 0, (
        "old points not cleaned"
    )
    print(f"  OK  reindex built v2 ({r.chunk_count} pts), cut over, cleaned v1")

    # --- failed reindex leaves old version active ---
    doc2 = DocumentState(
        document_id=document_id,
        filename="d.md",
        content_type="text/markdown",
        size=10,
        status=DocumentStatus.REINDEXING.value,
        document_version=1,
        checksum="x",
        original_object_key=okey,
        markdown_object_key=mkey,
        index_version=1,
    )
    store2 = FakeReindexStore(doc=doc2, configs={1: cfg1, 2: cfg2}, active_version=1)
    vectors2 = FakeVectorIndex()
    _seed_points(vectors2, document_id, index_version=1)

    class _BrokenEmbedding(FakeEmbedding):
        async def dense(self, texts):
            raise DomainError("embedding down", code=ErrorCode.EMBEDDING_TIMEOUT, retryable=True)

    p = ReindexPipeline(
        store=store2,
        object_store=s3,
        vector_index=vectors2,
        embedding=_BrokenEmbedding(DIMENSION),
        parser_registry=None,
        embedding_batch_size=16,
        consumer_name="reindex-worker",
        owner="w1",
        lease_ttl_seconds=120,
        domain_id=DOMAIN,
    )
    rf = await p.run(event)
    assert rf.status == "failed" and rf.retryable, rf
    assert store2.active_version == 1, "failed reindex must keep old version active"
    assert await vectors2.count_for_document(document_id, index_version=2) == 0, (
        "temp target not cleaned"
    )
    print("  OK  failed reindex kept v1 active and cleaned temp target points")

    # --- cancellation stops before cutover ---
    doc3 = DocumentState(
        document_id=document_id,
        filename="d.md",
        content_type="text/markdown",
        size=10,
        status=DocumentStatus.REINDEXING.value,
        document_version=1,
        checksum="x",
        original_object_key=okey,
        markdown_object_key=mkey,
        index_version=1,
    )
    store3 = FakeReindexStore(doc=doc3, configs={1: cfg1, 2: cfg2}, active_version=1)
    store3.job.cancel = True
    vectors3 = FakeVectorIndex()
    _seed_points(vectors3, document_id, index_version=1)
    pc = ReindexPipeline(
        store=store3,
        object_store=s3,
        vector_index=vectors3,
        embedding=FakeEmbedding(DIMENSION),
        parser_registry=None,
        embedding_batch_size=16,
        consumer_name="reindex-worker",
        owner="w1",
        lease_ttl_seconds=120,
        domain_id=DOMAIN,
    )
    rc = await pc.run(event)
    assert rc.status == "failed" and not rc.retryable, rc
    assert store3.active_version == 1
    print("  OK  cancellation stopped reindex before cutover")


async def run() -> None:
    await run_deletion()
    await run_reindex()
    print("lifecycle smoke: PASS")


if __name__ == "__main__":
    asyncio.run(run())
