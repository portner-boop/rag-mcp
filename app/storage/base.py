"""Storage backend protocols (bpmn-mcp `storage/base.py` convention).

Concrete adapters live beside this file — named after the backend, not generically:
`s3.py` (S3ObjectStore), `qdrant.py` (QdrantIndex), `embedding.py` (EmbeddingClient),
`postgres/` (engine + models + repositories), `queue/` (topology, publisher, relay).
Key/path policy is pure and IO-free in `keys.py` (the security boundary).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class BlobStore(Protocol):
    async def get_bytes(self, key: str) -> bytes: ...
    async def put_bytes(self, key: str, data: bytes, *, content_type: str) -> None: ...
    async def exists(self, key: str) -> bool: ...
