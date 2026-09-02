"""Deterministic rank fusion and deduplication (spec section 12.2 step 6).

Reciprocal Rank Fusion (RRF): score(id) = sum over lists of 1 / (k + rank), rank 1-based.
Fusion depends only on rank order, so it is fully deterministic; ties break on chunk id
so identical inputs always yield identical output order.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.storage.qdrant import VectorHit

DEFAULT_RRF_K = 60


@dataclass
class FusedHit:
    chunk_id: str
    score: float
    payload: dict


def reciprocal_rank_fusion(
    ranked_lists: list[list[VectorHit]], *, k: int = DEFAULT_RRF_K
) -> list[FusedHit]:
    scores: dict[str, float] = {}
    payloads: dict[str, dict] = {}
    for hits in ranked_lists:
        for rank, hit in enumerate(hits, start=1):
            scores[hit.id] = scores.get(hit.id, 0.0) + 1.0 / (k + rank)
            # Keep the first-seen payload for a chunk id (dedup by chunk id).
            payloads.setdefault(hit.id, hit.payload)
    fused = [
        FusedHit(chunk_id=cid, score=score, payload=payloads[cid]) for cid, score in scores.items()
    ]
    fused.sort(key=lambda h: (-h.score, h.chunk_id))
    return fused
