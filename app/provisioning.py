from __future__ import annotations

import structlog

from app.config import Settings
from app.shared.errors import ConfigurationError
from app.storage.postgres.engine import Database
from app.storage.postgres.repositories import IndexConfigRepository
from app.storage.qdrant import QdrantIndex

log = structlog.get_logger("provisioning")


def _verify_active_config(active, settings: Settings) -> None:
    mismatch = {
        field: {"active": current, "configured": configured}
        for field, current, configured in (
            ("dense_model", active.dense_model, settings.embedding_dense_model),
            (
                "dense_dimension",
                active.dense_dimension,
                settings.embedding_dense_dimension,
            ),
            ("qdrant_collection", active.qdrant_collection, settings.qdrant_collection),
        )
        if current != configured
    }
    if mismatch:
        raise ConfigurationError(
            "Active index config does not match the configured embedding setup; "
            "reindex (D04) or, in development, drop the collection and re-provision",
            details={"version": active.version, **mismatch},
        )


async def ensure_index_config(database: Database, settings: Settings) -> int:
    async with database.session() as session:
        repo = IndexConfigRepository(session)
        active = await repo.get_active()
        if active is not None:
            _verify_active_config(active, settings)
            return active.version
        existing = await repo.get_by_version(1)
        if existing is None:
            await repo.create(
                version=1,
                dense_model=settings.embedding_dense_model,
                dense_dimension=settings.embedding_dense_dimension,
                sparse_model=settings.embedding_sparse_model,
                reranker_model=settings.embedding_reranker_model,
                chunk_size_tokens=settings.chunk_size_tokens,
                chunk_overlap_tokens=settings.chunk_overlap_tokens,
                qdrant_collection=settings.qdrant_collection,
                active=False,
            )
        await repo.activate(1)
        log.info("provisioning.index_config.activated", version=1)
        return 1


async def ensure_qdrant(qdrant: QdrantIndex, settings: Settings) -> None:
    await qdrant.ensure_collection(
        dense_dimension=settings.embedding_dense_dimension,
        sparse_idf=settings.sparse_provider == "bm25",
    )


async def provision(database: Database, qdrant: QdrantIndex, settings: Settings) -> None:
    await ensure_qdrant(qdrant, settings)
    await ensure_index_config(database, settings)
