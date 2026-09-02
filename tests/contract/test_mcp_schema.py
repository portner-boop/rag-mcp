from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.shared.contracts.mcp import (
    PrepareUploadInput,
    SearchKnowledgeInput,
    SearchKnowledgeOutput,
)


def test_search_input_valid_defaults() -> None:
    payload = SearchKnowledgeInput(query="how is leave transferred")
    assert payload.limit == 10 and payload.max_candidates == 50
    assert payload.filters.document_version is None


def test_search_input_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        SearchKnowledgeInput(query="x", top_k=5)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"query": ""},
        {"query": "  "},
        {"query": "x", "limit": 0},
        {"query": "x", "limit": 51},
        {"query": "x", "limit": 10, "max_candidates": 5},
        {"query": "x", "max_candidates": 201},
    ],
)
def test_search_input_bounds(kwargs) -> None:
    with pytest.raises(ValidationError):
        SearchKnowledgeInput(**kwargs)


def test_search_output_roundtrip() -> None:
    out = SearchKnowledgeOutput(
        query_id="q",
        results=[],
        search_meta={
            "dense_candidates": 1,
            "sparse_candidates": 0,
            "reranked": False,
            "duration_ms": 5,
        },
    )
    assert SearchKnowledgeOutput.model_validate(out.model_dump()) == out


def test_prepare_upload_requires_positive_size() -> None:
    with pytest.raises(ValidationError):
        PrepareUploadInput(filename="f.pdf", content_type="application/pdf", size=0, created_by="u")
