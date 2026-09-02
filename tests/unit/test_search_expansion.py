"""Selective query expansion: trigger only on a low-confidence first pass (Tier 3.1)."""

from __future__ import annotations

import pytest

from app.ingestion.ports import IndexConfigState, PointData
from app.search.service import SearchService
from app.shared.contracts.mcp import SearchKnowledgeInput
from app.testing.embeddings import dense_vector, sparse_vector
from app.testing.fakes import FakeEmbedding, FakeExpander, FakeSearchStore, FakeVectorIndex

DIMENSION = 64


class _Settings:
    mcp_tool_timeout_seconds = 30.0
    search_rrf_k = 60
    search_max_chunk_chars = 4000
    enable_reranker = True
    allow_dense_only_fallback = True
    allow_sparse_only_fallback = False
    search_return_full_table = True
    embedding_query_instruction = ""
    enable_query_expansion = True
    query_expansion_min_rerank_score = 0.30
    query_expansion_max_variants = 3
    enable_hyde = True


def _seed(vectors: FakeVectorIndex, document_id: str, text: str) -> None:
    sv = sparse_vector(text)
    vectors.points[f"{document_id}:0"] = PointData(
        id=f"{document_id}:0",
        dense=dense_vector(text, DIMENSION),
        sparse_indices=sv.indices,
        sparse_values=sv.values,
        payload={
            "document_id": document_id,
            "filename": f"{document_id}.md",
            "text": text,
            "chunk_index": 0,
            "index_version": 1,
        },
    )


def _service(expander: FakeExpander, settings: _Settings) -> tuple[SearchService, FakeVectorIndex]:
    config = IndexConfigState(
        version=1,
        dense_model="fake",
        dense_dimension=DIMENSION,
        sparse_model="fake",
        qdrant_collection="c",
        reranker_model="fake",
    )
    vectors = FakeVectorIndex()
    _seed(vectors, "docA", "alpha beta gamma")
    store = FakeSearchStore(active_config=config, existing_ids={"docA"})
    service = SearchService(
        store=store,
        vectors=vectors,
        embedding=FakeEmbedding(DIMENSION),
        settings=settings,
        expander=expander,
    )
    return service, vectors


async def _search(service: SearchService, query: str):
    return await service.search_knowledge(
        SearchKnowledgeInput(query=query, limit=5, max_candidates=20)
    )


@pytest.mark.asyncio
async def test_low_confidence_first_pass_triggers_expansion() -> None:
    expander = FakeExpander(variants=["alpha beta"])
    service, _ = _service(expander, _Settings())
    # "zeta" shares no words with the only doc -> rerank score 0 < threshold -> expand.
    out = await _search(service, "zeta")
    assert out.search_meta.expanded is True
    assert expander.calls == 1


@pytest.mark.asyncio
async def test_confident_first_pass_does_not_expand() -> None:
    expander = FakeExpander(variants=["whatever"])
    service, _ = _service(expander, _Settings())
    # Query equals the doc text -> rerank score 1.0 >= threshold -> no expansion, no LLM call.
    out = await _search(service, "alpha beta gamma")
    assert out.search_meta.expanded is False
    assert expander.calls == 0


@pytest.mark.asyncio
async def test_flag_off_never_expands_even_on_low_score() -> None:
    settings = _Settings()
    settings.enable_query_expansion = False
    expander = FakeExpander(variants=["alpha beta"])
    service, _ = _service(expander, settings)
    out = await _search(service, "zeta")
    assert out.search_meta.expanded is False
    assert expander.calls == 0


@pytest.mark.asyncio
async def test_empty_expansion_degrades_quietly() -> None:
    expander = FakeExpander(variants=[])  # expander reachable but yields nothing
    service, _ = _service(expander, _Settings())
    out = await _search(service, "zeta")
    assert out.search_meta.expanded is False  # no extra lists -> not marked expanded
    assert expander.calls == 1  # it was consulted
    assert out.results  # the original result is still returned
