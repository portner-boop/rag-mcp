from __future__ import annotations

import uuid

POINT_NAMESPACE = uuid.UUID("6f1d0c9e-3f4a-5b6c-8d7e-0a1b2c3d4e5f")


def stable_point_id(
    document_id: str,
    document_version: int,
    index_version: int,
    chunk_index: int,
) -> str:
    name = f"{document_id}:{document_version}:{index_version}:{chunk_index}"
    return str(uuid.uuid5(POINT_NAMESPACE, name))


def new_uuid() -> str:
    return str(uuid.uuid4())
