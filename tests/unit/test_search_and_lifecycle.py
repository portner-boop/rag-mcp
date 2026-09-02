from __future__ import annotations

from app.search.fusion import reciprocal_rank_fusion
from app.search.normalize import normalize_query
from app.storage.qdrant import VectorHit


def test_normalize_preserves_intent() -> None:
    assert normalize_query("  How is   leave\ttransferred? ") == "How is leave transferred?"


def test_rrf_is_deterministic_and_dedupes() -> None:
    dense = [VectorHit("a", 0.9, {}), VectorHit("b", 0.8, {})]
    sparse = [VectorHit("b", 5.0, {}), VectorHit("c", 4.0, {})]
    fused = reciprocal_rank_fusion([dense, sparse])
    ids = [f.chunk_id for f in fused]
    assert ids[0] == "b"
    assert set(ids) == {"a", "b", "c"} and len(ids) == 3
    assert reciprocal_rank_fusion([dense, sparse]) == fused


def test_stable_point_id_changes_with_index_version() -> None:
    from app.shared.ids import stable_point_id

    assert stable_point_id("d", 1, 1, 0) != stable_point_id("d", 1, 2, 0)
    assert stable_point_id("d", 1, 2, 0) == stable_point_id("d", 1, 2, 0)
