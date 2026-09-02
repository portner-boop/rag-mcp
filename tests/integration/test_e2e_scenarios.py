"""End-to-end scenarios on in-memory fakes (spec section 18 required E2E 1-8).

These exercise the real ingestion/search/deletion/reindex engines wired together; only
the external stores/embeddings are faked. Real-infra (Testcontainers) variants live in
``test_real_infra.py`` (skipped unless RUN_INTEGRATION=1).
"""

from __future__ import annotations

from app.deletion.pipeline import DeletionPipeline
from app.ingestion.ports import DocumentState, IndexConfigState
from app.reindex.pipeline import ReindexPipeline
from app.shared.contracts.mcp import SearchKnowledgeInput
from app.shared.contracts.queue import DocumentDeletionRequested, DocumentReindexRequested
from app.shared.enums import DocumentStatus
from app.shared.errors import ErrorCode, UpstreamError
from app.shared.ids import new_uuid
from app.shared.time import to_rfc3339, utcnow
from app.storage.keys import markdown_key
from app.testing.fakes import FakeDeletionStore, FakeEmbedding, FakeReindexStore
from app.testing.harness import (
    DIMENSION,
    build_ingestion_pipeline,
    ingestion_setup,
    search_service,
)

POLICY = (
    "# HR Policy\n\n## Leave\n\n"
    "Unused annual leave is transferred to the next year up to a capped balance. "
    "Employees request the leave transfer through the HR portal before year end.\n\n"
    "## Remote work\n\nRemote work is available two days per week with manager approval.\n"
) * 2


async def _ingest_ready():
    setup = ingestion_setup(text=POLICY)
    result = await build_ingestion_pipeline(setup).run(setup.event)
    assert result.status == "completed"
    assert setup.store.docs[setup.document_id].state.status == DocumentStatus.READY.value
    return setup, result


# --- 1. ops ingestion -> READY -> chat search with citation metadata ------------------
async def test_scenario_1_ingest_then_search_with_citations() -> None:
    setup, result = await _ingest_ready()
    svc = search_service(setup.vectors, setup.embedding, existing_ids={setup.document_id})
    out = await svc.search_knowledge(
        SearchKnowledgeInput(query="how is unused leave transferred", limit=5)
    )
    assert out.results
    top = out.results[0]
    assert top.document_id == setup.document_id
    assert top.filename == "d.md" and top.chunk_id and top.text
    assert top.index_version == 1 and top.source_id == "S1"
    assert out.search_meta.duration_ms >= 0


# --- 2. duplicate ingestion delivery -> one effect ------------------------------------
async def test_scenario_2_duplicate_ingestion_one_effect() -> None:
    setup, first = await _ingest_ready()
    count1 = await setup.vectors.count_for_document(setup.document_id, index_version=1)
    second = await build_ingestion_pipeline(setup).run(setup.event)
    assert second.status == "duplicate"
    assert await setup.vectors.count_for_document(setup.document_id, index_version=1) == count1
    completed = [e for e in setup.store.outbox if e["event_type"] == "DocumentIngestionCompleted"]
    assert len(completed) == 1


# --- 3. embedding failure -> retry/recovery, never a false READY ----------------------
async def test_scenario_3_embedding_failure_then_recovery() -> None:
    setup = ingestion_setup(text=POLICY)

    async def boom(_texts):
        raise UpstreamError("embed down", code=ErrorCode.EMBEDDING_TIMEOUT, retryable=True)

    setup.embedding.dense = boom  # type: ignore[method-assign]
    failed = await build_ingestion_pipeline(setup).run(setup.event)
    assert failed.status == "failed" and failed.retryable
    assert setup.store.docs[setup.document_id].state.status != DocumentStatus.READY.value

    # Recover: healthy embedding, same event -> READY (no double effect, still idempotent).
    setup.embedding = FakeEmbedding(DIMENSION)
    recovered = await build_ingestion_pipeline(setup).run(setup.event)
    assert recovered.status == "completed"
    assert setup.store.docs[setup.document_id].state.status == DocumentStatus.READY.value


# --- 4. hybrid search -> bounded reranked result --------------------------------------
async def test_scenario_4_hybrid_bounded_reranked() -> None:
    setup, _ = await _ingest_ready()
    svc = search_service(setup.vectors, setup.embedding, existing_ids={setup.document_id})
    out = await svc.search_knowledge(SearchKnowledgeInput(query="leave transfer", limit=2))
    assert len(out.results) <= 2  # bounded by limit
    assert out.search_meta.reranked is True
    assert out.search_meta.dense_candidates > 0 and out.search_meta.sparse_candidates > 0


# --- 5. reranker unavailable -> declared fallback -------------------------------------
async def test_scenario_5_rerank_fallback_declared() -> None:
    setup, _ = await _ingest_ready()

    async def broken_rerank(*_a, **_k):
        raise UpstreamError("rerank down")

    setup.embedding.rerank = broken_rerank  # type: ignore[method-assign]
    svc = search_service(setup.vectors, setup.embedding, existing_ids={setup.document_id})
    out = await svc.search_knowledge(SearchKnowledgeInput(query="leave transfer", limit=5))
    assert out.results
    assert out.search_meta.reranked is False  # fallback declared in meta


# --- 6. delete -> immediate search exclusion -> complete cleanup ----------------------
async def test_scenario_6_delete_excludes_then_cleans_up() -> None:
    setup, _ = await _ingest_ready()
    doc_id = setup.document_id

    # Immediate exclusion: the document is marked excluded before its points are removed.
    svc = search_service(setup.vectors, setup.embedding, existing_ids={doc_id}, excluded={doc_id})
    out = await svc.search_knowledge(SearchKnowledgeInput(query="leave transfer", limit=5))
    assert all(r.document_id != doc_id for r in out.results)

    # Complete cleanup by the deletion pipeline.
    doc_state = setup.store.docs[doc_id].state
    doc_state.status = DocumentStatus.DELETING.value
    del_store = FakeDeletionStore(doc=doc_state)
    pipeline = DeletionPipeline(
        store=del_store,
        object_store=setup.s3,
        vector_index=setup.vectors,
        consumer_name="deletion-worker",
        owner="w",
        lease_ttl_seconds=120,
        domain_id="hr",
    )
    event = DocumentDeletionRequested(
        event_id=new_uuid(),
        occurred_at=to_rfc3339(utcnow()),
        domain="hr",
        document_id=doc_id,
        job_id="j",
        attempt=0,
        requested_by="admin",
    )
    result = await pipeline.run(event)
    assert result.status == "completed"
    assert await setup.vectors.count_all_for_document(doc_id) == 0
    assert del_store.doc.status == DocumentStatus.DELETED.value


def _reindex_scenario(setup, *, embedding):
    doc = DocumentState(
        document_id=setup.document_id,
        filename="d.md",
        content_type="text/markdown",
        size=10,
        status=DocumentStatus.REINDEXING.value,
        document_version=1,
        checksum="x",
        original_object_key=setup.original_key,
        markdown_object_key=markdown_key(setup.document_id),
        index_version=1,
    )
    cfg1 = IndexConfigState(
        version=1,
        dense_model="m",
        dense_dimension=DIMENSION,
        sparse_model="s",
        qdrant_collection="c",
        chunk_size_tokens=40,
        chunk_overlap_tokens=8,
    )
    cfg2 = IndexConfigState(
        version=2,
        dense_model="m",
        dense_dimension=DIMENSION,
        sparse_model="s",
        qdrant_collection="c",
        chunk_size_tokens=40,
        chunk_overlap_tokens=8,
    )
    store = FakeReindexStore(doc=doc, configs={1: cfg1, 2: cfg2}, active_version=1)
    pipeline = ReindexPipeline(
        store=store,
        object_store=setup.s3,
        vector_index=setup.vectors,
        embedding=embedding,
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
        document_id=setup.document_id,
        job_id="j",
        attempt=0,
        source_index_version=1,
        target_index_version=2,
        reason="new chunker",
    )
    return store, pipeline, event


# --- 7. reindex failure -> old version still searchable -------------------------------
async def test_scenario_7_reindex_failure_keeps_old_searchable() -> None:
    setup, _ = await _ingest_ready()

    class _Broken(FakeEmbedding):
        async def dense(self, texts):
            raise UpstreamError("down", code=ErrorCode.EMBEDDING_TIMEOUT, retryable=True)

    store, pipeline, event = _reindex_scenario(setup, embedding=_Broken(DIMENSION))
    result = await pipeline.run(event)
    assert result.status == "failed"
    assert store.active_version == 1  # old version stays active
    # Old version (1) still searchable.
    svc = search_service(
        setup.vectors, setup.embedding, active_version=1, existing_ids={setup.document_id}
    )
    out = await svc.search_knowledge(SearchKnowledgeInput(query="leave transfer", limit=5))
    assert any(r.document_id == setup.document_id for r in out.results)


# --- 8. reindex success -> atomic cutover -> old cleanup ------------------------------
async def test_scenario_8_reindex_success_cutover_and_cleanup() -> None:
    setup, _ = await _ingest_ready()
    store, pipeline, event = _reindex_scenario(setup, embedding=setup.embedding)
    result = await pipeline.run(event)
    assert result.status == "completed"
    assert store.active_version == 2  # cutover after verify
    assert (
        await setup.vectors.count_for_document(setup.document_id, index_version=1) == 0
    )  # old gone
    # New active version (2) is searchable.
    svc = search_service(
        setup.vectors, setup.embedding, active_version=2, existing_ids={setup.document_id}
    )
    out = await svc.search_knowledge(SearchKnowledgeInput(query="leave transfer", limit=5))
    assert any(r.document_id == setup.document_id and r.index_version == 2 for r in out.results)
