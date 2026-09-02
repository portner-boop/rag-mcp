from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class BlobStore(Protocol):
    async def get_bytes(self, key: str) -> bytes: ...
    async def put_bytes(self, key: str, data: bytes, *, content_type: str) -> None: ...
    async def exists(self, key: str) -> bool: ...
