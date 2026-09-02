from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from qdrant_client import AsyncQdrantClient, models

from app.shared.errors import ConfigurationError, UpstreamError

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"

_PAYLOAD_INDEXES: dict[str, models.PayloadSchemaType] = {
    "document_id": models.PayloadSchemaType.KEYWORD,
    "document_version": models.PayloadSchemaType.INTEGER,
    "index_version": models.PayloadSchemaType.INTEGER,
    "content_type": models.PayloadSchemaType.KEYWORD,
    "created_at": models.PayloadSchemaType.KEYWORD,
}


@dataclass
class QdrantPoint:
    id: str
    dense: list[float]
    sparse_indices: list[int]
    sparse_values: list[float]
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class VectorHit:
    id: str
    score: float
    payload: dict[str, Any] = field(default_factory=dict)


class QdrantIndex:
    def __init__(
        self, *, url: str, api_key: str | None, collection: str, timeout: float = 10.0
    ) -> None:
        self._client = AsyncQdrantClient(
            url=url, api_key=api_key or None, timeout=timeout, check_compatibility=False
        )
        self._collection = collection

    @property
    def collection(self) -> str:
        return self._collection

    async def close(self) -> None:
        await self._client.close()

    async def ensure_collection(self, *, dense_dimension: int, sparse_idf: bool = False) -> None:
        sparse_params = models.SparseVectorParams(
            index=models.SparseIndexParams(),
            modifier=models.Modifier.IDF if sparse_idf else None,
        )
        try:
            exists = await self._client.collection_exists(self._collection)
            if exists:
                await self._verify_collection(dense_dimension, sparse_params)
            else:
                await self._client.create_collection(
                    collection_name=self._collection,
                    vectors_config={
                        DENSE_VECTOR: models.VectorParams(
                            size=dense_dimension, distance=models.Distance.COSINE
                        )
                    },
                    sparse_vectors_config={SPARSE_VECTOR: sparse_params},
                )
            for name, schema in _PAYLOAD_INDEXES.items():
                try:
                    await self._client.create_payload_index(
                        collection_name=self._collection,
                        field_name=name,
                        field_schema=schema,
                    )
                except Exception:  # noqa: BLE001 - index may already exist
                    pass
        except ConfigurationError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise UpstreamError("Qdrant ensure_collection failed") from exc

    async def validate_schema(self, *, dense_dimension: int) -> None:
        try:
            info = await self._client.get_collection(self._collection)
        except Exception as exc:  # noqa: BLE001
            raise ConfigurationError(
                "Qdrant collection is not reachable or missing",
                details={"collection": self._collection},
            ) from exc

        vectors = info.config.params.vectors
        dense_params = vectors.get(DENSE_VECTOR) if isinstance(vectors, dict) else None
        if dense_params is None:
            raise ConfigurationError(
                "Qdrant collection is missing the named 'dense' vector",
                details={"collection": self._collection},
            )
        if dense_params.size != dense_dimension:
            raise ConfigurationError(
                "Qdrant dense dimension does not match active index config",
                details={
                    "collection": self._collection,
                    "expected": dense_dimension,
                    "actual": dense_params.size,
                },
            )
        sparse = info.config.params.sparse_vectors or {}
        if SPARSE_VECTOR not in sparse:
            raise ConfigurationError(
                "Qdrant collection is missing the named 'sparse' vector",
                details={"collection": self._collection},
            )

    async def _verify_collection(
        self, dense_dimension: int, sparse_params: models.SparseVectorParams
    ) -> None:
        await self.validate_schema(dense_dimension=dense_dimension)
        info = await self._client.get_collection(self._collection)
        current = (info.config.params.sparse_vectors or {}).get(SPARSE_VECTOR)
        if current is not None and current.modifier != sparse_params.modifier:
            await self._client.update_collection(
                collection_name=self._collection,
                sparse_vectors_config={SPARSE_VECTOR: sparse_params},
            )

    async def drop_collection(self) -> None:
        await self._client.delete_collection(self._collection)

    async def upsert(self, points: list[QdrantPoint]) -> None:
        if not points:
            return
        qdrant_points = [
            models.PointStruct(
                id=p.id,
                vector={
                    DENSE_VECTOR: p.dense,
                    SPARSE_VECTOR: models.SparseVector(
                        indices=p.sparse_indices, values=p.sparse_values
                    ),
                },
                payload=p.payload,
            )
            for p in points
        ]
        try:
            await self._client.upsert(
                collection_name=self._collection, points=qdrant_points, wait=True
            )
        except Exception as exc:  # noqa: BLE001
            raise UpstreamError("Qdrant upsert failed") from exc

    async def count_for_document(self, document_id: str, *, index_version: int) -> int:
        flt = models.Filter(
            must=[
                models.FieldCondition(
                    key="document_id", match=models.MatchValue(value=document_id)
                ),
                models.FieldCondition(
                    key="index_version", match=models.MatchValue(value=index_version)
                ),
            ]
        )
        return await self._count(flt)

    async def count_all_for_document(self, document_id: str) -> int:
        flt = models.Filter(
            must=[
                models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))
            ]
        )
        return await self._count(flt)

    async def _count(self, flt: models.Filter) -> int:
        try:
            result = await self._client.count(
                collection_name=self._collection, count_filter=flt, exact=True
            )
        except Exception as exc:  # noqa: BLE001
            raise UpstreamError("Qdrant count failed") from exc
        return result.count

    def _search_filter(
        self,
        *,
        index_version: int,
        document_ids: list[str] | None,
        document_version: int | None,
        created_from: str | None,
        created_to: str | None,
        exclude_document_ids: list[str] | None,
    ) -> models.Filter:
        must: list[models.Condition] = [
            models.FieldCondition(key="index_version", match=models.MatchValue(value=index_version))
        ]
        if document_ids:
            must.append(
                models.FieldCondition(
                    key="document_id", match=models.MatchAny(any=list(document_ids))
                )
            )
        if document_version is not None:
            must.append(
                models.FieldCondition(
                    key="document_version", match=models.MatchValue(value=document_version)
                )
            )
        if created_from or created_to:
            must.append(
                models.FieldCondition(
                    key="created_at",
                    range=models.DatetimeRange(gte=created_from, lte=created_to),
                )
            )
        must_not: list[models.Condition] = []
        if exclude_document_ids:
            must_not.append(
                models.FieldCondition(
                    key="document_id", match=models.MatchAny(any=list(exclude_document_ids))
                )
            )
        return models.Filter(must=must, must_not=must_not or None)

    @staticmethod
    def _to_hits(scored: list[models.ScoredPoint]) -> list[VectorHit]:
        return [
            VectorHit(id=str(p.id), score=float(p.score), payload=p.payload or {}) for p in scored
        ]

    async def dense_search(
        self,
        *,
        vector: list[float],
        limit: int,
        index_version: int,
        document_ids: list[str] | None = None,
        document_version: int | None = None,
        created_from: str | None = None,
        created_to: str | None = None,
        exclude_document_ids: list[str] | None = None,
    ) -> list[VectorHit]:
        flt = self._search_filter(
            index_version=index_version,
            document_ids=document_ids,
            document_version=document_version,
            created_from=created_from,
            created_to=created_to,
            exclude_document_ids=exclude_document_ids,
        )
        try:
            response = await self._client.query_points(
                collection_name=self._collection,
                query=vector,
                using=DENSE_VECTOR,
                query_filter=flt,
                limit=limit,
                with_payload=True,
            )
        except Exception as exc:  # noqa: BLE001
            raise UpstreamError("Qdrant dense search failed") from exc
        return self._to_hits(response.points)

    async def sparse_search(
        self,
        *,
        indices: list[int],
        values: list[float],
        limit: int,
        index_version: int,
        document_ids: list[str] | None = None,
        document_version: int | None = None,
        created_from: str | None = None,
        created_to: str | None = None,
        exclude_document_ids: list[str] | None = None,
    ) -> list[VectorHit]:
        flt = self._search_filter(
            index_version=index_version,
            document_ids=document_ids,
            document_version=document_version,
            created_from=created_from,
            created_to=created_to,
            exclude_document_ids=exclude_document_ids,
        )
        try:
            response = await self._client.query_points(
                collection_name=self._collection,
                query=models.SparseVector(indices=indices, values=values),
                using=SPARSE_VECTOR,
                query_filter=flt,
                limit=limit,
                with_payload=True,
            )
        except Exception as exc:  # noqa: BLE001
            raise UpstreamError("Qdrant sparse search failed") from exc
        return self._to_hits(response.points)

    async def retrieve(self, ids: list[str]) -> list[models.Record]:
        try:
            return await self._client.retrieve(
                collection_name=self._collection, ids=ids, with_payload=True
            )
        except Exception as exc:  # noqa: BLE001
            raise UpstreamError("Qdrant retrieve failed") from exc

    async def delete_document(self, document_id: str, *, index_version: int) -> None:
        flt = models.Filter(
            must=[
                models.FieldCondition(
                    key="document_id", match=models.MatchValue(value=document_id)
                ),
                models.FieldCondition(
                    key="index_version", match=models.MatchValue(value=index_version)
                ),
            ]
        )
        await self._delete_by_filter(flt)

    async def delete_document_all(self, document_id: str) -> None:
        flt = models.Filter(
            must=[
                models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))
            ]
        )
        await self._delete_by_filter(flt)

    async def _delete_by_filter(self, flt: models.Filter) -> None:
        try:
            await self._client.delete(
                collection_name=self._collection,
                points_selector=models.FilterSelector(filter=flt),
                wait=True,
            )
        except Exception as exc:  # noqa: BLE001
            raise UpstreamError("Qdrant delete failed") from exc
