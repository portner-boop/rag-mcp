from __future__ import annotations

import pytest

from app.ingestion.ports import IndexConfigState, PointData
from app.search.service import SearchService
from app.shared.contracts.mcp import SearchKnowledgeInput
from app.testing.embeddings import dense_vector, sparse_vector
from app.testing.fakes import FakeEmbedding, FakeSearchStore, FakeVectorIndex

DIMENSION = 64


class _Settings:
    mcp_tool_timeout_seconds = 30.0
    search_rrf_k = 60
    search_max_chunk_chars = 4000
    enable_reranker = False
    allow_dense_only_fallback = True
    allow_sparse_only_fallback = False
    search_return_full_table = True
    embedding_query_instruction = ""
    enable_query_expansion = False
    search_min_dense_confidence = 0.6
    search_min_rerank_confidence = 0.45


def _service(settings):
    config = IndexConfigState(
        version=1,
        dense_model="f",
        dense_dimension=DIMENSION,
        sparse_model="f",
        qdrant_collection="c",
        reranker_model="f",
    )
    vectors = FakeVectorIndex()
    sv = sparse_vector("alpha beta gamma")
    vectors.points["d:0"] = PointData(
        id="d:0",
        dense=dense_vector("alpha beta gamma", DIMENSION),
        sparse_indices=sv.indices,
        sparse_values=sv.values,
        payload={
            "document_id": "d",
            "filename": "d.md",
            "text": "alpha beta gamma",
            "chunk_index": 0,
            "index_version": 1,
        },
    )
    store = FakeSearchStore(active_config=config, existing_ids={"d"})
    return SearchService(
        store=store, vectors=vectors, embedding=FakeEmbedding(DIMENSION), settings=settings
    )


async def _search(svc):
    return await svc.search_knowledge(
        SearchKnowledgeInput(query="alpha", limit=5, max_candidates=20)
    )


@pytest.mark.asyncio
async def test_top_dense_score_is_surfaced() -> None:
    out = await _search(_service(_Settings()))
    assert out.search_meta.top_dense_score is not None


@pytest.mark.asyncio
async def test_low_confidence_when_top_dense_below_threshold() -> None:
    settings = _Settings()
    settings.search_min_dense_confidence = 1.1  # nothing can exceed -> always low confidence
    out = await _search(_service(settings))
    assert out.search_meta.low_confidence is True


@pytest.mark.asyncio
async def test_confident_when_threshold_is_low() -> None:
    settings = _Settings()
    settings.search_min_dense_confidence = -1.0  # everything clears -> confident
    out = await _search(_service(settings))
    assert out.search_meta.low_confidence is False


@pytest.mark.asyncio
async def test_no_rerank_score_without_reranker() -> None:
    out = await _search(_service(_Settings()))
    assert out.search_meta.top_rerank_score is None


# FakeEmbedding.rerank scores by token Jaccard: "alpha" vs "alpha beta gamma" -> 1/3.


@pytest.mark.asyncio
async def test_rerank_score_governs_when_reranked() -> None:
    settings = _Settings()
    settings.enable_reranker = True
    settings.search_min_dense_confidence = -1.0  # dense would say confident
    settings.search_min_rerank_confidence = 0.5  # rerank (1/3) says low
    out = await _search(_service(settings))
    assert out.search_meta.reranked is True
    assert out.search_meta.top_rerank_score == pytest.approx(1 / 3)
    assert out.search_meta.low_confidence is True


@pytest.mark.asyncio
async def test_rerank_confidence_overrides_dense_fallback() -> None:
    settings = _Settings()
    settings.enable_reranker = True
    settings.search_min_dense_confidence = 1.1  # dense would say low
    settings.search_min_rerank_confidence = 0.2  # rerank (1/3) clears
    out = await _search(_service(settings))
    assert out.search_meta.reranked is True
    assert out.search_meta.low_confidence is False
