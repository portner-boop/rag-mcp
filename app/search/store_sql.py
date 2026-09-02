from __future__ import annotations

from app.ingestion.ports import IndexConfigState
from app.storage.postgres.engine import Database
from app.storage.postgres.repositories import DocumentRepository, IndexConfigRepository


class SqlSearchStore:
    def __init__(self, database: Database) -> None:
        self._db = database

    async def get_active_index_config(self) -> IndexConfigState:
        async with self._db.session() as session:
            cfg = await IndexConfigRepository(session).get_active_or_raise()
            return IndexConfigState(
                version=cfg.version,
                dense_model=cfg.dense_model,
                dense_dimension=cfg.dense_dimension,
                sparse_model=cfg.sparse_model,
                qdrant_collection=cfg.qdrant_collection,
                reranker_model=cfg.reranker_model,
                chunk_size_tokens=cfg.chunk_size_tokens,
                chunk_overlap_tokens=cfg.chunk_overlap_tokens,
            )

    async def document_exists(self, document_id: str) -> bool:
        async with self._db.session() as session:
            return await DocumentRepository(session).get(document_id) is not None

    async def excluded_document_ids(self) -> list[str]:
        async with self._db.session() as session:
            return await DocumentRepository(session).excluded_ids()
