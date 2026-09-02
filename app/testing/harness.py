"""Test/smoke harness: builders that wire the fakes into runnable pipelines and events.

Used by the D05 test suite and the smoke scripts to keep scenario setup terse.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.ports import DocumentState, IndexConfigState
from app.shared.contracts.queue import DocumentIngestionRequested
from app.shared.enums import DocumentStatus
from app.shared.ids import new_uuid
from app.shared.time import to_rfc3339, utcnow
from app.storage.keys import markdown_key, original_key
from app.testing.fakes import FakeEmbedding, FakeIngestionStore, FakeObjectStore, FakeVectorIndex

DIMENSION = 64
DOMAIN = "hr"

CHAT_TOKEN = "chat-test-token"
OPS_TOKEN = "ops-test-token"


def make_settings(**overrides):
    """A valid Settings for wiring a Container in tests (no live infra is contacted)."""
    from app.config import Settings
    from app.session.auth import hash_token

    base = dict(
        domain_id=DOMAIN,
        postgres_url="postgresql+asyncpg://u:p@localhost/hr",
        qdrant_url="http://localhost:6333",
        qdrant_collection="hr-knowledge",
        s3_endpoint="http://localhost:9000",
        s3_bucket="hr-documents",
        s3_access_key="k",
        s3_secret_key="s",
        chat_service_token_hash=hash_token(CHAT_TOKEN),
        ops_service_token_hash=hash_token(OPS_TOKEN),
        embedding_api_url="http://localhost:8000",
        rabbitmq_url="amqp://localhost",
        queue_namespace=DOMAIN,
    )
    base.update(overrides)
    return Settings(**base)


def make_container(settings=None):
    from app.container import Container

    return Container(settings or make_settings())


def index_config(*, version: int = 1, dimension: int = DIMENSION) -> IndexConfigState:
    return IndexConfigState(
        version=version,
        dense_model="fake-dense",
        dense_dimension=dimension,
        sparse_model="fake-sparse",
        qdrant_collection="hr-knowledge",
        reranker_model="fake-rerank",
        chunk_size_tokens=40,
        chunk_overlap_tokens=8,
    )


@dataclass
class IngestionSetup:
    store: FakeIngestionStore
    s3: FakeObjectStore
    vectors: FakeVectorIndex
    embedding: FakeEmbedding
    document_id: str
    job_id: str
    event: DocumentIngestionRequested
    original_key: str


def make_ingestion_event(
    document_id: str,
    job_id: str,
    okey: str,
    *,
    domain: str = DOMAIN,
    index_version: int = 1,
    attempt: int = 0,
) -> DocumentIngestionRequested:
    return DocumentIngestionRequested(
        event_id=new_uuid(),
        occurred_at=to_rfc3339(utcnow()),
        domain=domain,
        document_id=document_id,
        job_id=job_id,
        attempt=attempt,
        trace_id="0" * 32,
        original_object_key=okey,
        index_version=index_version,
    )


def ingestion_setup(
    *,
    text: str,
    content_type: str = "text/markdown",
    filename: str = "d.md",
    dimension: int = DIMENSION,
) -> IngestionSetup:
    document_id, job_id = new_uuid(), new_uuid()
    okey = original_key(document_id, filename)
    data = text.encode("utf-8")
    store = FakeIngestionStore(active_config=index_config(dimension=dimension))
    store.seed_document(
        DocumentState(
            document_id=document_id,
            filename=filename,
            content_type=content_type,
            size=len(data),
            status=DocumentStatus.QUEUED.value,
            document_version=1,
            checksum=None,
            original_object_key=okey,
            markdown_object_key=None,
        )
    )
    store.seed_job(job_id, document_id)
    s3 = FakeObjectStore()
    s3.objects[okey] = (data, content_type)
    return IngestionSetup(
        store=store,
        s3=s3,
        vectors=FakeVectorIndex(),
        embedding=FakeEmbedding(dimension),
        document_id=document_id,
        job_id=job_id,
        event=make_ingestion_event(document_id, job_id, okey),
        original_key=okey,
    )


def build_ingestion_pipeline(
    setup: IngestionSetup,
    *,
    chunk_size: int = 40,
    overlap: int = 8,
    batch: int = 16,
    owner: str = "w1",
    domain: str = DOMAIN,
) -> IngestionPipeline:
    return IngestionPipeline(
        store=setup.store,
        object_store=setup.s3,
        vector_index=setup.vectors,
        embedding=setup.embedding,
        parser_registry=_parser_registry(),
        markdown_key_for=markdown_key,
        chunk_size_tokens=chunk_size,
        chunk_overlap_tokens=overlap,
        embedding_batch_size=batch,
        consumer_name="ingestion-worker",
        owner=owner,
        lease_ttl_seconds=120,
        domain_id=domain,
    )


def _parser_registry():
    from app.ingestion.parser.base import default_registry

    return default_registry()


class SearchSettings:
    mcp_tool_timeout_seconds = 30.0
    search_dense_candidates = 50
    search_sparse_candidates = 50
    search_rrf_k = 60
    search_max_chunk_chars = 4000
    enable_reranker = True
    allow_dense_only_fallback = True
    allow_sparse_only_fallback = False


def search_service(
    vectors: FakeVectorIndex,
    embedding: FakeEmbedding,
    *,
    active_version: int = 1,
    existing_ids=(),
    excluded=(),
    settings=None,
):
    from app.search.service import SearchService
    from app.testing.fakes import FakeSearchStore

    store = FakeSearchStore(
        active_config=index_config(version=active_version),
        existing_ids=set(existing_ids),
        excluded=set(excluded),
    )
    return SearchService(
        store=store, vectors=vectors, embedding=embedding, settings=settings or SearchSettings()
    )


@dataclass
class FakeMessage:
    """Minimal stand-in for an aio-pika incoming message (records ack/nack/reject)."""

    body: bytes
    headers: dict = field(default_factory=dict)
    acked: bool = False
    nacked: bool = False
    rejected: bool = False
    requeue: bool | None = None

    @classmethod
    def from_event(cls, event) -> FakeMessage:  # noqa: ANN001
        return cls(body=json.dumps(event.model_dump(mode="json")).encode("utf-8"))

    async def ack(self) -> None:
        self.acked = True

    async def nack(self, requeue: bool = False) -> None:
        self.nacked = True
        self.requeue = requeue

    async def reject(self, requeue: bool = False) -> None:
        self.rejected = True
        self.requeue = requeue
