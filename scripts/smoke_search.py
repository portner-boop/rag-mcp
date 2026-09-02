"""Smoke check: production hybrid search on in-memory fakes (D03 checklist).

Proves: an original query returns ranked relevant chunks; dense+sparse fuse and rerank;
search_meta reflects the rerank fallback; the active index version is enforced;
deleting/deleted and non-active-version items are excluded; document_id filter validates.

Run: uv run python scripts/smoke_search.py
"""

from __future__ import annotations

import asyncio

from app.ingestion.ports import IndexConfigState, PointData
from app.search.service import SearchService
from app.shared.contracts.mcp import SearchFilters, SearchKnowledgeInput
from app.shared.errors import ValidationError
from app.shared.ids import new_uuid
from app.shared.time import to_rfc3339, utcnow
from app.storage.embedding import EmbeddingValidationError
from app.testing.embeddings import dense_vector, sparse_vector
from app.testing.fakes import FakeEmbedding, FakeSearchStore, FakeVectorIndex

DIMENSION = 64


class _Settings:
    mcp_tool_timeout_seconds = 30.0
    search_dense_candidates = 50
    search_sparse_candidates = 50
    search_rrf_k = 60
    search_max_chunk_chars = 4000
    enable_reranker = True
    allow_dense_only_fallback = True
    allow_sparse_only_fallback = False


def _seed_point(
    vectors: FakeVectorIndex,
    *,
    document_id: str,
    chunk_index: int,
    text: str,
    index_version: int = 1,
) -> str:
    pid = f"{document_id}:{chunk_index}:{index_version}"
    sv = sparse_vector(text)
    vectors.points[pid] = PointData(
        id=pid,
        dense=dense_vector(text, DIMENSION),
        sparse_indices=sv.indices,
        sparse_values=sv.values,
        payload={
            "document_id": document_id,
            "filename": f"{document_id}.md",
            "text": text,
            "chunk_index": chunk_index,
            "document_version": 1,
            "index_version": index_version,
            "content_type": "text/markdown",
            "created_at": to_rfc3339(utcnow()),
        },
    )
    return pid


async def run() -> None:
    print("hybrid search smoke:")
    config = IndexConfigState(
        version=2,
        dense_model="fake-dense",
        dense_dimension=DIMENSION,
        sparse_model="fake-sparse",
        qdrant_collection="hr-knowledge",
        reranker_model="fake-rerank",
    )
    vectors = FakeVectorIndex()
    doc_leave, doc_remote, doc_stale, doc_deleting = (new_uuid() for _ in range(4))
    _seed_point(
        vectors,
        document_id=doc_leave,
        chunk_index=0,
        text="Unused annual leave is transferred to the next year up to a cap.",
        index_version=2,
    )
    _seed_point(
        vectors,
        document_id=doc_remote,
        chunk_index=0,
        text="Remote work is available two days per week with manager approval.",
        index_version=2,
    )
    # old index version -> must be excluded (active is v2)
    _seed_point(
        vectors,
        document_id=doc_stale,
        chunk_index=0,
        text="Leave transfer under the old policy version.",
        index_version=1,
    )
    # a deleting document at the active version -> must be excluded via status filter
    _seed_point(
        vectors,
        document_id=doc_deleting,
        chunk_index=0,
        text="Leave transfer draft pending deletion.",
        index_version=2,
    )

    store = FakeSearchStore(
        active_config=config,
        existing_ids={doc_leave, doc_remote, doc_stale, doc_deleting},
        excluded={doc_deleting},
    )
    embedding = FakeEmbedding(DIMENSION)
    service = SearchService(store=store, vectors=vectors, embedding=embedding, settings=_Settings())

    # --- 1. ranked relevant chunks; active version enforced; exclusions applied ---
    out = await service.search_knowledge(
        SearchKnowledgeInput(query="How is unused leave transferred?", limit=5, max_candidates=20)
    )
    ids = [r.document_id for r in out.results]
    assert out.results, "search returned nothing"
    assert out.results[0].document_id == doc_leave, ids
    assert doc_stale not in ids, "old index version leaked into results"
    assert doc_deleting not in ids, "deleting document leaked into results"
    assert out.search_meta.reranked is True
    assert out.search_meta.dense_candidates > 0 and out.search_meta.sparse_candidates > 0
    assert all(r.index_version == 2 for r in out.results)
    print(
        f"  OK  ranked hybrid+rerank result; top={out.results[0].document_id[:8]} "
        f"(dense={out.search_meta.dense_candidates}, sparse={out.search_meta.sparse_candidates})"
    )

    # --- 2. rerank fallback -> reranked=false but still returns hybrid top-N ---
    async def _broken_rerank(*_a, **_k):
        raise EmbeddingValidationError("rerank down")

    embedding.rerank = _broken_rerank  # type: ignore[method-assign]
    out2 = await service.search_knowledge(
        SearchKnowledgeInput(query="unused leave transferred", limit=5, max_candidates=20)
    )
    assert out2.results, "fallback returned nothing"
    assert out2.search_meta.reranked is False
    print("  OK  rerank fallback -> reranked=false, hybrid top-N preserved")

    # --- 3. dense-only fallback when sparse embedding is unavailable ---
    async def _broken_sparse(_texts):
        raise EmbeddingValidationError("sparse down")

    embedding.sparse = _broken_sparse  # type: ignore[method-assign]
    out3 = await service.search_knowledge(
        SearchKnowledgeInput(query="leave transfer", limit=5, max_candidates=20)
    )
    assert out3.search_meta.sparse_candidates == 0 and out3.search_meta.dense_candidates > 0
    print("  OK  dense-only fallback (sparse unavailable) -> sparse_candidates=0")

    # --- 4. unknown document id in filter is rejected ---
    try:
        await service.search_knowledge(
            SearchKnowledgeInput(
                query="x", limit=3, document_ids=["does-not-exist"], filters=SearchFilters()
            )
        )
    except ValidationError:
        print("  OK  unknown document_id filter rejected")
    else:
        raise AssertionError("expected ValidationError for unknown document id")

    print("search smoke: PASS")


if __name__ == "__main__":
    asyncio.run(run())
