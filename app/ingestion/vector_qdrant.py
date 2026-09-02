"""Adapter binding the ``VectorIndexPort`` to the shared Qdrant client."""

from __future__ import annotations

from app.ingestion.ports import PointData
from app.storage.qdrant import QdrantIndex, QdrantPoint


class QdrantVectorIndex:
    def __init__(self, qdrant: QdrantIndex) -> None:
        self._qdrant = qdrant

    async def upsert(self, points: list[PointData]) -> None:
        await self._qdrant.upsert(
            [
                QdrantPoint(
                    id=p.id,
                    dense=p.dense,
                    sparse_indices=p.sparse_indices,
                    sparse_values=p.sparse_values,
                    payload=p.payload,
                )
                for p in points
            ]
        )

    async def count_for_document(self, document_id: str, *, index_version: int) -> int:
        return await self._qdrant.count_for_document(document_id, index_version=index_version)

    async def delete_document(self, document_id: str, *, index_version: int) -> None:
        await self._qdrant.delete_document(document_id, index_version=index_version)

    async def delete_document_all(self, document_id: str) -> None:
        await self._qdrant.delete_document_all(document_id)

    async def count_all_for_document(self, document_id: str) -> int:
        return await self._qdrant.count_all_for_document(document_id)

    async def retrieve_ids(self, ids: list[str]) -> list[str]:
        records = await self._qdrant.retrieve(ids)
        return [str(r.id) for r in records]
