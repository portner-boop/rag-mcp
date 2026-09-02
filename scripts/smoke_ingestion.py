from __future__ import annotations

import asyncio

from app.ingestion.parser.base import default_registry
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.ports import DocumentState, IndexConfigState
from app.shared.contracts.queue import DocumentIngestionRequested
from app.shared.enums import DocumentStatus, EventType, JobStatus
from app.shared.ids import new_uuid
from app.shared.time import to_rfc3339, utcnow
from app.storage.keys import markdown_key, original_key
from app.testing.embeddings import dense_vector
from app.testing.fakes import (
    FakeEmbedding,
    FakeIngestionStore,
    FakeObjectStore,
    FakeVectorIndex,
)

DIMENSION = 64
DOMAIN = "hr"

DOCUMENT_MD = (
    """# HR Policy

## Leave

Annual leave is accrued monthly. Unused leave is transferred to the next year up to a
capped balance. Employees request leave transfer through the HR portal before year end.

## Remote work

Remote work is available two days per week subject to manager approval.
"""
    * 3
)


def _new_pipeline(store, s3, vectors, embedding, owner="w1"):
    return IngestionPipeline(
        store=store,
        object_store=s3,
        vector_index=vectors,
        embedding=embedding,
        parser_registry=default_registry(),
        markdown_key_for=markdown_key,
        chunk_size_tokens=40,
        chunk_overlap_tokens=8,
        embedding_batch_size=16,
        consumer_name="ingestion-worker",
        owner=owner,
        lease_ttl_seconds=120,
        domain_id=DOMAIN,
    )


async def run() -> None:
    document_id = new_uuid()
    job_id = new_uuid()
    okey = original_key(document_id, "hr-policy.md")

    config = IndexConfigState(
        version=1,
        dense_model="fake-dense",
        dense_dimension=DIMENSION,
        sparse_model="fake-sparse",
        qdrant_collection="hr-knowledge",
    )
    store = FakeIngestionStore(active_config=config)
    store.seed_document(
        DocumentState(
            document_id=document_id,
            filename="hr-policy.md",
            content_type="text/markdown",
            size=len(DOCUMENT_MD.encode()),
            status=DocumentStatus.QUEUED.value,
            document_version=1,
            checksum=None,
            original_object_key=okey,
            markdown_object_key=None,
        )
    )
    store.seed_job(job_id, document_id)

    s3 = FakeObjectStore()
    s3.objects[okey] = (DOCUMENT_MD.encode("utf-8"), "text/markdown")
    vectors = FakeVectorIndex()
    embedding = FakeEmbedding(DIMENSION)

    event = DocumentIngestionRequested(
        event_id=new_uuid(),
        occurred_at=to_rfc3339(utcnow()),
        domain=DOMAIN,
        document_id=document_id,
        job_id=job_id,
        attempt=0,
        trace_id="0" * 32,
        original_object_key=okey,
        index_version=1,
    )

    print("ingestion pipeline smoke:")

    result = await _new_pipeline(store, s3, vectors, embedding).run(event)
    assert result.status == "completed", result
    assert result.chunk_count > 0
    doc = store.docs[document_id].state
    assert doc.status == DocumentStatus.READY.value, doc.status
    assert doc.chunk_count == result.chunk_count
    point_count = await vectors.count_for_document(document_id, index_version=1)
    assert point_count == result.chunk_count, (point_count, result.chunk_count)
    assert markdown_key(document_id) in s3.objects, "markdown must be stored in S3"
    assert store.jobs[job_id].status == JobStatus.COMPLETED.value
    completed = [
        e for e in store.outbox if e["event_type"] == EventType.DOCUMENT_INGESTION_COMPLETED.value
    ]
    assert len(completed) == 1, "exactly one completed event"
    print(f"  OK  READY after verified upsert: {result.chunk_count} chunks / {point_count} points")

    result2 = await _new_pipeline(store, s3, vectors, embedding).run(event)
    assert result2.status == "duplicate", result2
    assert await vectors.count_for_document(document_id, index_version=1) == point_count
    assert (
        len(
            [
                e
                for e in store.outbox
                if e["event_type"] == EventType.DOCUMENT_INGESTION_COMPLETED.value
            ]
        )
        == 1
    )
    print("  OK  repeated delivery produced no double effect (inbox idempotency)")

    query_vec = dense_vector("How is unused leave transferred?", DIMENSION)
    hits = await vectors.dense_search(vector=query_vec, limit=3, index_version=1)
    assert hits, "dense search returned no points"
    top_payload = hits[0].payload
    assert top_payload["document_id"] == document_id
    assert "leave" in top_payload["text"].lower()
    print(f"  OK  dense query retrieved an ingested chunk (score={hits[0].score:.3f})")

    await _run_failures()

    print("ingestion smoke: PASS")


def _seed_single(store, s3, *, content_type: str, data: bytes):
    document_id = new_uuid()
    job_id = new_uuid()
    okey = original_key(document_id, "doc.bin")
    store.seed_document(
        DocumentState(
            document_id=document_id,
            filename="doc.bin",
            content_type=content_type,
            size=len(data),
            status=DocumentStatus.QUEUED.value,
            document_version=1,
            checksum=None,
            original_object_key=okey,
            markdown_object_key=None,
        )
    )
    store.seed_job(job_id, document_id)
    s3.objects[okey] = (data, content_type)
    event = DocumentIngestionRequested(
        event_id=new_uuid(),
        occurred_at=to_rfc3339(utcnow()),
        domain=DOMAIN,
        document_id=document_id,
        job_id=job_id,
        attempt=0,
        trace_id="0" * 32,
        original_object_key=okey,
        index_version=1,
    )
    return document_id, job_id, event


async def _run_failures() -> None:
    config = IndexConfigState(
        version=1,
        dense_model="fake-dense",
        dense_dimension=DIMENSION,
        sparse_model="fake-sparse",
        qdrant_collection="hr-knowledge",
    )

    store = FakeIngestionStore(active_config=config)
    s3, vectors, embedding = FakeObjectStore(), FakeVectorIndex(), FakeEmbedding(DIMENSION)
    document_id, _job_id, event = _seed_single(
        store, s3, content_type="application/zip", data=b"PK\x03\x04binary"
    )
    result = await _new_pipeline(store, s3, vectors, embedding).run(event)
    assert result.status == "failed" and not result.retryable, result
    assert result.error_code == "UNSUPPORTED_FILE", result.error_code
    assert await vectors.count_for_document(document_id, index_version=1) == 0
    print(f"  OK  unsupported file fails terminally at {result.stage} (no points written)")

    store = FakeIngestionStore(active_config=config)
    s3, vectors, embedding = FakeObjectStore(), FakeVectorIndex(), FakeEmbedding(DIMENSION)
    document_id, job_id, event = _seed_single(
        store, s3, content_type="text/markdown", data=b"# Doc\n\nsome content here"
    )
    store.jobs[job_id].cancel = True
    result = await _new_pipeline(store, s3, vectors, embedding).run(event)
    assert result.status == "failed" and not result.retryable, result
    assert store.docs[document_id].state.status != DocumentStatus.READY.value
    print("  OK  cancellation stops the pipeline before READY")


if __name__ == "__main__":
    asyncio.run(run())
