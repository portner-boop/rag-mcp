"""Shared pytest fixtures + Hypothesis profile for the D05 suite (spec section 18)."""

from __future__ import annotations

import pytest
from hypothesis import settings

from app.ingestion.ports import IndexConfigState
from app.testing.fakes import FakeEmbedding, FakeIngestionStore, FakeObjectStore, FakeVectorIndex

DIMENSION = 64

settings.register_profile("default", deadline=None, max_examples=100)
settings.load_profile("default")


@pytest.fixture
def index_config() -> IndexConfigState:
    return IndexConfigState(
        version=1,
        dense_model="fake-dense",
        dense_dimension=DIMENSION,
        sparse_model="fake-sparse",
        qdrant_collection="hr-knowledge",
        reranker_model="fake-rerank",
        chunk_size_tokens=40,
        chunk_overlap_tokens=8,
    )


@pytest.fixture
def fakes(index_config: IndexConfigState):
    return {
        "store": FakeIngestionStore(active_config=index_config),
        "s3": FakeObjectStore(),
        "vectors": FakeVectorIndex(),
        "embedding": FakeEmbedding(DIMENSION),
    }
