"""Stable identifier derivation (spec invariant 9).

A Qdrant point ID must be deterministically derived from
document/version/index/chunk identity so that re-running ingestion upserts the same
points instead of creating duplicates.
"""

from __future__ import annotations

import uuid

# Fixed namespace UUID for this stack. Do not change: it anchors all stable IDs.
POINT_NAMESPACE = uuid.UUID("6f1d0c9e-3f4a-5b6c-8d7e-0a1b2c3d4e5f")


def stable_point_id(
    document_id: str,
    document_version: int,
    index_version: int,
    chunk_index: int,
) -> str:
    """Deterministic point/chunk UUID for a given chunk of a given document version."""
    name = f"{document_id}:{document_version}:{index_version}:{chunk_index}"
    return str(uuid.uuid5(POINT_NAMESPACE, name))


def new_uuid() -> str:
    return str(uuid.uuid4())
