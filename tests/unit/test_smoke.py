from __future__ import annotations

import pytest

from app.config import _scoped_to_domain
from app.ingestion.chunker import DeterministicChunker


def test_isolation_guard() -> None:
    assert _scoped_to_domain("hr-documents", "hr")
    assert _scoped_to_domain("domain.hr.ingestion", "hr")
    assert _scoped_to_domain("hr", "hr")
    assert not _scoped_to_domain("warehouse-documents", "hr")
    assert not _scoped_to_domain("sharior", "hr")


def test_chunker_is_deterministic() -> None:
    md = "# HR Policy\n\n" + ("word " * 60) + "\n## Leave\n\n" + ("leave " * 60)
    chunker = DeterministicChunker(chunk_size_tokens=40, chunk_overlap_tokens=10)
    a = chunker.chunk(md)
    b = chunker.chunk(md)
    assert [c.text for c in a] == [c.text for c in b]
    assert [c.chunk_index for c in a] == list(range(len(a)))
    assert any("HR Policy" in c.section_path for c in a)


@pytest.mark.asyncio
async def test_ingestion_pipeline_reaches_ready(fakes) -> None:
    from app.ingestion.parser.base import default_registry
    from app.ingestion.pipeline import IngestionPipeline
    from app.ingestion.ports import DocumentState
    from app.shared.contracts.queue import DocumentIngestionRequested
    from app.shared.enums import DocumentStatus
    from app.shared.ids import new_uuid
    from app.shared.time import to_rfc3339, utcnow
    from app.storage.keys import markdown_key, original_key

    document_id, job_id = new_uuid(), new_uuid()
    okey = original_key(document_id, "policy.md")
    fakes["store"].seed_document(
        DocumentState(
            document_id=document_id,
            filename="policy.md",
            content_type="text/markdown",
            size=64,
            status=DocumentStatus.QUEUED.value,
            document_version=1,
            checksum=None,
            original_object_key=okey,
            markdown_object_key=None,
        )
    )
    fakes["store"].seed_job(job_id, document_id)
    fakes["s3"].objects[okey] = (b"# Doc\n\n" + b"leave transfer policy " * 30, "text/markdown")

    pipeline = IngestionPipeline(
        store=fakes["store"],
        object_store=fakes["s3"],
        vector_index=fakes["vectors"],
        embedding=fakes["embedding"],
        parser_registry=default_registry(),
        markdown_key_for=markdown_key,
        chunk_size_tokens=40,
        chunk_overlap_tokens=8,
        embedding_batch_size=16,
        consumer_name="ingestion-worker",
        owner="w1",
        lease_ttl_seconds=120,
        domain_id="hr",
    )
    event = DocumentIngestionRequested(
        event_id=new_uuid(),
        occurred_at=to_rfc3339(utcnow()),
        domain="hr",
        document_id=document_id,
        job_id=job_id,
        attempt=0,
        original_object_key=okey,
        index_version=1,
    )
    result = await pipeline.run(event)
    assert result.status == "completed"
    assert fakes["store"].docs[document_id].state.status == DocumentStatus.READY.value
    assert (
        await fakes["vectors"].count_for_document(document_id, index_version=1)
        == result.chunk_count
    )
