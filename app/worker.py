"""Worker mode: wire adapters, the pipeline factory and the queue consumer, then run.

The worker consumes the ingestion queue with manual acknowledgements (spec sections 10,
11). It shares the same storage adapters and ingestion engine as the server; only the
transport differs.
"""

from __future__ import annotations

import asyncio
import signal

import structlog

from app.config import Settings
from app.container import build_embedding, get_settings
from app.deletion.consumer import DeletionConsumer
from app.deletion.pipeline import DeletionPipeline
from app.deletion.store_sql import SqlDeletionStore
from app.ingestion.consumer import IngestionConsumer
from app.ingestion.parser.base import default_registry
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.store_sql import SqlIngestionStore
from app.ingestion.vector_qdrant import QdrantVectorIndex
from app.reindex.consumer import ReindexConsumer
from app.reindex.pipeline import ReindexPipeline
from app.reindex.store_sql import SqlReindexStore
from app.shared.ids import new_uuid
from app.storage.keys import markdown_key
from app.storage.postgres.engine import Database
from app.storage.qdrant import QdrantIndex
from app.storage.queue import connection as rmq
from app.storage.queue.publisher import Publisher
from app.storage.queue.topology import declare_topology
from app.storage.s3 import S3ObjectStore
from app.worker_support.recovery import RecoveryService

log = structlog.get_logger("worker")

CONSUMER_NAME = "ingestion-worker"


class WorkerContainer:
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
        # Same factory as the server: documents must be embedded by exactly the model
        # queries are embedded with, or retrieval compares vectors from two spaces.
        self.embedding = build_embedding(settings)
        self.vector_index = QdrantVectorIndex(self.qdrant)
        self.parser_registry = default_registry()
        events_rk = settings.routing_key("events")
        self.store = SqlIngestionStore(database=self.database, events_routing_key=events_rk)
        self.deletion_store = SqlDeletionStore(database=self.database, events_routing_key=events_rk)
        self.reindex_store = SqlReindexStore(database=self.database, events_routing_key=events_rk)

    def make_pipeline(self, owner: str) -> IngestionPipeline:
        s = self.settings
        return IngestionPipeline(
            store=self.store,
            object_store=self.object_store,
            vector_index=self.vector_index,
            embedding=self.embedding,
            parser_registry=self.parser_registry,
            markdown_key_for=markdown_key,
            chunk_size_tokens=s.chunk_size_tokens,
            chunk_overlap_tokens=s.chunk_overlap_tokens,
            embedding_batch_size=s.embedding_batch_size,
            consumer_name=CONSUMER_NAME,
            owner=owner,
            lease_ttl_seconds=s.lease_ttl_seconds,
            domain_id=s.domain_id,
        )

    def make_consumer(self) -> IngestionConsumer:
        s = self.settings
        owner = f"{CONSUMER_NAME}-{new_uuid()[:8]}"
        return IngestionConsumer(
            pipeline_factory=self.make_pipeline,
            store=self.store,
            domain_id=s.domain_id,
            consumer_name=CONSUMER_NAME,
            max_attempts=s.max_attempts,
            retry_schedule=s.retry_schedule_seconds,
            retry_jitter=s.retry_jitter_seconds,
            owner=owner,
        )

    def make_deletion_consumer(self) -> DeletionConsumer:
        s = self.settings
        pipeline = DeletionPipeline(
            store=self.deletion_store,
            object_store=self.object_store,
            vector_index=self.vector_index,
            consumer_name="deletion-worker",
            owner=f"deletion-worker-{new_uuid()[:8]}",
            lease_ttl_seconds=s.lease_ttl_seconds,
            domain_id=s.domain_id,
        )
        return DeletionConsumer(
            pipeline=pipeline,
            store=self.deletion_store,
            domain_id=s.domain_id,
            max_attempts=s.max_attempts,
            retry_schedule=s.retry_schedule_seconds,
            retry_jitter=s.retry_jitter_seconds,
        )

    def make_reindex_consumer(self) -> ReindexConsumer:
        s = self.settings
        pipeline = ReindexPipeline(
            store=self.reindex_store,
            object_store=self.object_store,
            vector_index=self.vector_index,
            embedding=self.embedding,
            parser_registry=self.parser_registry,
            embedding_batch_size=s.embedding_batch_size,
            consumer_name="reindex-worker",
            owner=f"reindex-worker-{new_uuid()[:8]}",
            lease_ttl_seconds=s.lease_ttl_seconds,
            domain_id=s.domain_id,
        )
        return ReindexConsumer(
            pipeline=pipeline,
            store=self.reindex_store,
            domain_id=s.domain_id,
            max_attempts=s.max_attempts,
            retry_schedule=s.retry_schedule_seconds,
            retry_jitter=s.retry_jitter_seconds,
        )

    async def shutdown(self) -> None:
        await self.embedding.aclose()
        await self.qdrant.close()
        await self.database.dispose()


def build_worker_container(settings: Settings | None = None) -> WorkerContainer:
    return WorkerContainer(settings or get_settings())


async def _recovery_loop(
    recovery: RecoveryService, *, interval: float, stop: asyncio.Event
) -> None:
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval)
        except TimeoutError:
            try:
                await recovery.run_once()
            except Exception:  # noqa: BLE001
                log.exception("recovery.error")


async def run(settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    container = build_worker_container(settings)

    connection = await rmq.connect(settings.rabbitmq_url)
    channel = await rmq.open_channel(connection, prefetch=settings.worker_prefetch)
    topology = await declare_topology(
        channel,
        exchange_name=settings.exchange_name,
        dlx_name=settings.dlx_name,
        dlq_name=settings.dlq_name,
        queue_name=settings.queue_name,
        routing_key=settings.routing_key,
    )

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:  # pragma: no cover - non-posix
            pass

    # Consume all three command queues with manual ack (spec sections 10, 11).
    await topology.queues["ingestion"].consume(container.make_consumer().handle, no_ack=False)
    await topology.queues["deletion"].consume(
        container.make_deletion_consumer().handle, no_ack=False
    )
    await topology.queues["reindex"].consume(container.make_reindex_consumer().handle, no_ack=False)

    # Stale upload/job/lease recovery loop (D04).
    recovery = RecoveryService(
        database=container.database, publisher=Publisher(topology.exchange), settings=settings
    )
    recovery_task = asyncio.create_task(
        _recovery_loop(recovery, interval=settings.recovery_interval_seconds, stop=stop)
    )

    log.info("worker.consuming", queues=["ingestion", "deletion", "reindex"])
    await stop.wait()

    log.info("worker.stopping")
    recovery_task.cancel()
    await connection.close()
    await container.shutdown()
