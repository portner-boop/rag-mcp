"""Composition root: build adapters and application services from configuration.

Fail-fast startup (spec section 5, 6): configuration is validated by ``Settings``; the
container additionally validates that the live Qdrant collection matches the active index
config (D01 completion checklist). Chat readiness and operational health are separated
(spec section 6): an S3/RabbitMQ/worker outage must not hide an otherwise searchable
index.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import structlog

from app.config import Settings, get_settings
from app.operational.dlq import DlqService
from app.operational.documents import DocumentReadService
from app.operational.downloads import DownloadService
from app.operational.ingestion import IngestionService
from app.operational.lifecycle import LifecycleService
from app.search.service import SearchService
from app.search.store_sql import SqlSearchStore
from app.session.auth import TokenAuthenticator
from app.storage.bm25 import Bm25Encoder
from app.storage.embedding import EmbeddingClient
from app.storage.embedding_openai import OpenAICompatibleEmbedding
from app.storage.generation_openai import OpenAICompatibleExpander
from app.storage.postgres.engine import Database
from app.storage.postgres.repositories import IndexConfigRepository
from app.storage.qdrant import QdrantIndex
from app.storage.queue import connection as rmq
from app.storage.queue.outbox_relay import OutboxRelay
from app.storage.queue.publisher import Publisher
from app.storage.queue.topology import declare_topology
from app.storage.s3 import S3ObjectStore

log = structlog.get_logger("bootstrap")


@dataclass
class HealthReport:
    ok: bool
    checks: dict[str, str] = field(default_factory=dict)


def build_embedding(settings: Settings):
    """Pick the retrieval backends configured for this deployment (spec section 13).

    ``openai``: dense + rerank over an OpenAI/Cohere-shaped gateway (OpenRouter), lexical
    branch computed locally as BM25 with Qdrant applying IDF.
    ``custom``: this stack's own Embedding API for all three, optionally with the same
    local BM25 in place of a served sparse model.
    """
    if settings.embedding_provider == "openai":
        return OpenAICompatibleEmbedding(
            base_url=settings.embedding_api_url,
            token=settings.embedding_api_token,
            dense_model=settings.embedding_dense_model,
            reranker_model=settings.embedding_reranker_model,
            dense_dimension=settings.embedding_dense_dimension,
            sparse_encoder=build_sparse_encoder(settings),
            timeout=settings.embedding_timeout_seconds,
            rerank_timeout=settings.rerank_timeout_seconds,
        )
    client = EmbeddingClient(
        base_url=settings.embedding_api_url,
        token=settings.embedding_api_token,
        dense_model=settings.embedding_dense_model,
        sparse_model=settings.embedding_sparse_model,
        reranker_model=settings.embedding_reranker_model,
        dense_dimension=settings.embedding_dense_dimension,
        timeout=settings.embedding_timeout_seconds,
        rerank_timeout=settings.rerank_timeout_seconds,
    )
    if settings.sparse_provider == "bm25":
        client.use_local_sparse(build_sparse_encoder(settings))
    return client


def build_expander(settings: Settings):
    """The selective query expander (Tier 3.1), or None when the feature is off.

    Uses the same OpenAI-shaped gateway as embeddings for chat completions; built only when
    enabled so no generation client exists on the common (expansion-off) deployment.
    """
    if not settings.enable_query_expansion:
        return None
    return OpenAICompatibleExpander(
        base_url=settings.embedding_api_url,
        token=settings.embedding_api_token,
        model=settings.expansion_model,
        timeout=settings.expansion_timeout_seconds,
    )


def build_sparse_encoder(settings: Settings) -> Bm25Encoder:
    return Bm25Encoder(
        k1=settings.bm25_k1,
        b=settings.bm25_b,
        avg_doc_len=settings.bm25_avg_doc_len_tokens,
        vocab_size=settings.bm25_vocab_size,
    )


class Container:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.database = Database(settings.postgres_url)
        self.object_store = S3ObjectStore(
            endpoint=settings.s3_endpoint,
            region=settings.s3_region,
            bucket=settings.s3_bucket,
            access_key=settings.s3_access_key,
            secret_key=settings.s3_secret_key,
            use_path_style=settings.s3_use_path_style,
        )
        self.qdrant = QdrantIndex(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            collection=settings.qdrant_collection,
            timeout=settings.qdrant_timeout_seconds,
        )
        self.embedding = build_embedding(settings)
        self.expander = build_expander(settings)
        self.authenticator = TokenAuthenticator(
            chat_hashes=settings.chat_service_token_hash,
            ops_hashes=settings.ops_service_token_hash,
        )

        # Services.
        self.search_service = SearchService(
            store=SqlSearchStore(self.database),
            vectors=self.qdrant,
            embedding=self.embedding,
            settings=settings,
            expander=self.expander,
        )
        self.document_service = DocumentReadService(self.database)
        self.ingestion_service = IngestionService(
            database=self.database, object_store=self.object_store, settings=settings
        )
        self.download_service = DownloadService(
            database=self.database, object_store=self.object_store, settings=settings
        )
        self.lifecycle_service = LifecycleService(database=self.database, settings=settings)

        # Queue (set on startup).
        self._connection = None
        self.publisher: Publisher | None = None
        self.dlq_service: DlqService | None = None
        self._relay: OutboxRelay | None = None
        self._relay_task: asyncio.Task | None = None

    async def startup(self, *, run_relay: bool = True) -> None:
        settings = self.settings
        await self.database.ping()
        active = await self._active_config_or_raise()
        # Validate the live Qdrant collection matches the active config (D01 checklist).
        await self.qdrant.validate_schema(dense_dimension=active.dense_dimension)

        # Operational plane: queue topology + outbox relay.
        self._connection = await rmq.connect(settings.rabbitmq_url)
        channel = await rmq.open_channel(self._connection)
        topology = await declare_topology(
            channel,
            exchange_name=settings.exchange_name,
            dlx_name=settings.dlx_name,
            dlq_name=settings.dlq_name,
            queue_name=settings.queue_name,
            routing_key=settings.routing_key,
        )
        self.publisher = Publisher(topology.exchange)
        self.dlq_service = DlqService(
            channel=channel,
            exchange=topology.exchange,
            dlq_name=settings.dlq_name,
            routing_key=settings.routing_key,
        )
        if run_relay:
            self._relay = OutboxRelay(database=self.database, publisher=self.publisher)
            self._relay_task = asyncio.create_task(self._relay.run())
        log.info("server.startup.complete", domain=settings.domain_id)

    async def shutdown(self) -> None:
        if self._relay is not None:
            self._relay.stop()
        if self._relay_task is not None:
            self._relay_task.cancel()
            try:
                await self._relay_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        if self._connection is not None:
            await self._connection.close()
        await self.embedding.aclose()
        if self.expander is not None:
            await self.expander.aclose()
        await self.qdrant.close()
        await self.database.dispose()

    async def _active_config_or_raise(self):
        async with self.database.session() as session:
            return await IndexConfigRepository(session).get_active_or_raise()

    async def chat_readiness(self) -> HealthReport:
        """Chat readiness: PostgreSQL metadata, Qdrant schema, active index config."""
        checks: dict[str, str] = {}
        ok = True
        try:
            active = await self._active_config_or_raise()
            checks["postgres"] = "ok"
            checks["index_config"] = f"v{active.version}"
        except Exception:  # noqa: BLE001
            ok = False
            checks["postgres"] = "fail"
        try:
            active = await self._active_config_or_raise()
            await self.qdrant.validate_schema(dense_dimension=active.dense_dimension)
            checks["qdrant"] = "ok"
        except Exception:  # noqa: BLE001
            ok = False
            checks["qdrant"] = "fail"
        return HealthReport(ok=ok, checks=checks)

    async def operational_health(self) -> HealthReport:
        """Operational health: adds S3 and RabbitMQ, which do not affect chat search."""
        report = await self.chat_readiness()
        checks = dict(report.checks)
        ok = report.ok
        try:
            await self.object_store.head("__healthcheck__/never")
            checks["s3"] = "ok"
        except Exception as exc:  # noqa: BLE001
            # NotFound is fine: it proves the bucket is reachable.
            checks["s3"] = "ok" if type(exc).__name__ == "NotFoundError" else "fail"
        checks["rabbitmq"] = (
            "ok" if self._connection is not None and not self._connection.is_closed else "fail"
        )
        return HealthReport(ok=ok, checks=checks)


def build_container(settings: Settings | None = None) -> Container:
    settings = settings or get_settings()
    settings.require_server_tokens()
    return Container(settings)


# --- process-wide container accessor --------------------------------------------------
# Tools are registered on `public_mcp` at import time, before the container exists; they
# fetch it lazily at call time via this accessor (set during server startup).

_CONTAINER: Container | None = None


def set_container(container: Container) -> None:
    global _CONTAINER
    _CONTAINER = container


def get_container() -> Container:
    if _CONTAINER is None:
        raise RuntimeError("Container is not initialized")
    return _CONTAINER
