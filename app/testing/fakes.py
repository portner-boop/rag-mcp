"""In-memory fakes implementing the ingestion pipeline ports (spec section 18)."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.ingestion.ports import (
    DocumentState,
    FinalizeData,
    IndexConfigState,
    PointData,
)
from app.shared.contracts.embedding import (
    RerankDocument,
    RerankResponse,
    RerankResult,
    SparseVector,
)
from app.shared.enums import DocumentStatus, JobStatus
from app.shared.errors import NotFoundError
from app.storage.qdrant import VectorHit
from app.testing.embeddings import dense_vector, rerank_score, sparse_vector


class FakeObjectStore:
    def __init__(self) -> None:
        self.objects: dict[str, tuple[bytes, str]] = {}

    async def get_bytes(self, key: str) -> bytes:
        if key not in self.objects:
            raise NotFoundError("Object not found", details={"key": key})
        return self.objects[key][0]

    async def put_bytes(self, key: str, data: bytes, *, content_type: str) -> None:
        self.objects[key] = (data, content_type)

    async def exists(self, key: str) -> bool:
        return key in self.objects

    async def delete(self, key: str) -> None:
        self.objects.pop(key, None)


class FakeVectorIndex:
    def __init__(self) -> None:
        self.points: dict[str, PointData] = {}

    async def upsert(self, points: list[PointData]) -> None:
        for p in points:
            self.points[p.id] = p

    async def count_for_document(self, document_id: str, *, index_version: int) -> int:
        return sum(
            1
            for p in self.points.values()
            if p.payload.get("document_id") == document_id
            and p.payload.get("index_version") == index_version
        )

    async def delete_document(self, document_id: str, *, index_version: int) -> None:
        self.points = {
            pid: p
            for pid, p in self.points.items()
            if not (
                p.payload.get("document_id") == document_id
                and p.payload.get("index_version") == index_version
            )
        }

    async def delete_document_all(self, document_id: str) -> None:
        self.points = {
            pid: p for pid, p in self.points.items() if p.payload.get("document_id") != document_id
        }

    async def count_all_for_document(self, document_id: str) -> int:
        return sum(1 for p in self.points.values() if p.payload.get("document_id") == document_id)

    async def retrieve_ids(self, ids: list[str]) -> list[str]:
        return [i for i in ids if i in self.points]

    def _matches(
        self, p: PointData, *, index_version, document_ids, document_version, exclude_document_ids
    ) -> bool:  # noqa: ANN001
        pd = p.payload
        if pd.get("index_version") != index_version:
            return False
        if document_ids and pd.get("document_id") not in document_ids:
            return False
        if document_version is not None and pd.get("document_version") != document_version:
            return False
        if exclude_document_ids and pd.get("document_id") in exclude_document_ids:
            return False
        return True

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
        def cos(a: list[float], b: list[float]) -> float:
            return sum(x * y for x, y in zip(a, b, strict=False))

        scored = [
            VectorHit(id=pid, score=cos(vector, p.dense), payload=p.payload)
            for pid, p in self.points.items()
            if self._matches(
                p,
                index_version=index_version,
                document_ids=document_ids,
                document_version=document_version,
                exclude_document_ids=exclude_document_ids,
            )
        ]
        scored.sort(key=lambda h: (-h.score, h.id))
        return scored[:limit]

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
        query = dict(zip(indices, values, strict=True))
        hits: list[VectorHit] = []
        for pid, p in self.points.items():
            if not self._matches(
                p,
                index_version=index_version,
                document_ids=document_ids,
                document_version=document_version,
                exclude_document_ids=exclude_document_ids,
            ):
                continue
            score = sum(
                query[i] * v
                for i, v in zip(p.sparse_indices, p.sparse_values, strict=True)
                if i in query
            )
            if score > 0:
                hits.append(VectorHit(id=pid, score=score, payload=p.payload))
        hits.sort(key=lambda h: (-h.score, h.id))
        return hits[:limit]


class FakeEmbedding:
    def __init__(self, dimension: int) -> None:
        self.dimension = dimension

    async def dense(self, texts: list[str]) -> list[list[float]]:
        return [dense_vector(t, self.dimension) for t in texts]

    async def sparse(self, texts: list[str]) -> list[SparseVector]:
        return [sparse_vector(t) for t in texts]

    async def rerank(
        self, query: str, documents: list[RerankDocument], *, top_n: int
    ) -> RerankResponse:
        scored = [RerankResult(id=d.id, score=rerank_score(query, d.text)) for d in documents]
        scored.sort(key=lambda r: -r.score)
        return RerankResponse(results=scored[:top_n])


@dataclass
class FakeExpander:
    """Canned query-expansion for search tests; records how many times it was invoked."""

    variants: list[str] = field(default_factory=list)
    calls: int = 0

    async def expand(self, query: str, *, num_variants: int, hyde: bool) -> list[str]:
        self.calls += 1
        return list(self.variants)[: num_variants + (1 if hyde else 0)]


@dataclass
class FakeSearchStore:
    active_config: IndexConfigState
    existing_ids: set[str] = field(default_factory=set)
    excluded: set[str] = field(default_factory=set)

    async def get_active_index_config(self) -> IndexConfigState:
        return self.active_config

    async def document_exists(self, document_id: str) -> bool:
        return document_id in self.existing_ids

    async def excluded_document_ids(self) -> list[str]:
        return list(self.excluded)


# --------------------------------------------------------------------------------------
# In-memory IngestionStore
# --------------------------------------------------------------------------------------


@dataclass
class _Doc:
    state: DocumentState
    indexed: bool = False
    error_code: str | None = None


@dataclass
class _Job:
    job_id: str
    document_id: str
    status: str = JobStatus.QUEUED.value
    stage: str | None = None
    progress: int = 0
    attempt: int = 0
    cancel: bool = False
    lease_owner: str | None = None


@dataclass
class FakeIngestionStore:
    active_config: IndexConfigState
    docs: dict[str, _Doc] = field(default_factory=dict)
    jobs: dict[str, _Job] = field(default_factory=dict)
    inbox: set[tuple[str, str]] = field(default_factory=set)
    outbox: list[dict] = field(default_factory=list)
    audit: list[dict] = field(default_factory=list)

    # --- seeding helpers ---
    def seed_document(self, state: DocumentState) -> None:
        self.docs[state.document_id] = _Doc(state=state)

    def seed_job(self, job_id: str, document_id: str) -> None:
        self.jobs[job_id] = _Job(job_id=job_id, document_id=document_id)

    # --- port methods ---
    async def inbox_seen(self, consumer: str, event_id: str) -> bool:
        return (consumer, event_id) in self.inbox

    async def get_document(self, document_id: str) -> DocumentState:
        if document_id not in self.docs:
            raise NotFoundError("Document not found")
        return self.docs[document_id].state

    async def get_active_index_config(self) -> IndexConfigState:
        return self.active_config

    async def find_ready_by_checksum(self, checksum: str, document_version: int) -> str | None:
        for doc in self.docs.values():
            if (
                doc.state.checksum == checksum
                and doc.state.document_version == document_version
                and doc.state.status == DocumentStatus.READY.value
            ):
                return doc.state.document_id
        return None

    async def begin_processing(
        self, job_id: str, document_id: str, *, owner: str, lease_ttl_seconds: int
    ) -> None:
        self.docs[document_id].state.status = DocumentStatus.PROCESSING.value
        job = self.jobs[job_id]
        job.status = JobStatus.PROCESSING.value
        job.lease_owner = owner

    async def set_stage(self, job_id: str, *, stage: str, progress: int) -> None:
        job = self.jobs[job_id]
        job.stage = stage
        job.progress = progress

    async def heartbeat(self, job_id: str, *, owner: str, lease_ttl_seconds: int) -> bool:
        return True

    async def cancel_requested(self, job_id: str) -> bool:
        return self.jobs[job_id].cancel

    async def persist_markdown_key(self, document_id: str, key: str) -> None:
        self.docs[document_id].state.markdown_object_key = key

    async def finalize_ready(
        self, *, document_id: str, job_id: str, consumer: str, event_id: str, data: FinalizeData
    ) -> None:
        doc = self.docs[document_id]
        doc.state.status = DocumentStatus.READY.value
        doc.state.checksum = data.checksum
        doc.state.chunk_count = data.chunk_count
        doc.state.index_version = data.index_version
        doc.indexed = True
        job = self.jobs[job_id]
        job.status = JobStatus.COMPLETED.value
        job.progress = 100
        self.outbox.append(data.completed_event)
        self.audit.append({"type": "completed", "document_id": document_id})
        self.inbox.add((consumer, event_id))

    async def complete_idempotent(self, *, job_id: str, consumer: str, event_id: str) -> None:
        self.jobs[job_id].status = JobStatus.COMPLETED.value
        self.jobs[job_id].progress = 100
        self.inbox.add((consumer, event_id))

    async def mark_failed(
        self,
        *,
        document_id: str,
        job_id: str,
        stage: str,
        error_code: str,
        error_message: str,
        attempt: int,
        set_document_failed: bool,
        failed_event: dict | None,
        consumer: str,
        event_id: str,
    ) -> None:
        job = self.jobs[job_id]
        job.stage = stage
        job.attempt = attempt
        if set_document_failed:
            doc = self.docs[document_id]
            doc.state.status = DocumentStatus.FAILED.value
            doc.error_code = error_code
            job.status = JobStatus.FAILED.value
            if failed_event is not None:
                failed_event.pop("_dead_letter", None)
                self.outbox.append(failed_event)
            self.inbox.add((consumer, event_id))
        else:
            job.status = JobStatus.RETRY_WAIT.value

    async def record_retry(self, job_id: str, *, attempt: int, available_in: float) -> None:
        job = self.jobs[job_id]
        job.status = JobStatus.RETRY_WAIT.value
        job.attempt = attempt


# --------------------------------------------------------------------------------------
# In-memory DeletionStore / ReindexStore
# --------------------------------------------------------------------------------------


@dataclass
class FakeDeletionStore:
    doc: DocumentState
    job: _Job = field(default_factory=lambda: _Job("j", "d"))
    inbox: set = field(default_factory=set)
    outbox: list = field(default_factory=list)

    async def inbox_seen(self, consumer, event_id):
        return (consumer, event_id) in self.inbox

    async def get_document(self, document_id):
        return self.doc

    async def begin_processing(self, job_id, document_id, *, owner, lease_ttl_seconds):
        self.doc.status = DocumentStatus.DELETING.value
        self.job.status = JobStatus.PROCESSING.value

    async def set_stage(self, job_id, *, stage, progress):
        self.job.stage = stage

    async def heartbeat(self, job_id, *, owner, lease_ttl_seconds):
        return True

    async def finalize_deleted(self, *, document_id, job_id, consumer, event_id, deleted_event):
        self.doc.status = DocumentStatus.DELETED.value
        self.job.status = JobStatus.COMPLETED.value
        self.outbox.append(deleted_event)
        self.inbox.add((consumer, event_id))

    async def mark_failed(
        self, *, document_id, job_id, set_document_failed, consumer, event_id, **_
    ):
        if set_document_failed:
            self.doc.status = DocumentStatus.DELETE_FAILED.value
            self.inbox.add((consumer, event_id))
        else:
            self.job.status = JobStatus.RETRY_WAIT.value

    async def record_retry(self, job_id, *, attempt, available_in):
        self.job.status = JobStatus.RETRY_WAIT.value


@dataclass
class FakeReindexStore:
    doc: DocumentState
    configs: dict  # version -> IndexConfigState
    active_version: int
    job: _Job = field(default_factory=lambda: _Job("j", "d"))
    inbox: set = field(default_factory=set)
    outbox: list = field(default_factory=list)

    async def inbox_seen(self, consumer, event_id):
        return (consumer, event_id) in self.inbox

    async def get_document(self, document_id):
        return self.doc

    async def get_index_config(self, version):
        return self.configs[version]

    async def begin_processing(self, job_id, document_id, *, owner, lease_ttl_seconds):
        self.job.status = JobStatus.PROCESSING.value

    async def set_stage(self, job_id, *, stage, progress):
        self.job.stage = stage

    async def heartbeat(self, job_id, *, owner, lease_ttl_seconds):
        return True

    async def cancel_requested(self, job_id):
        return self.job.cancel

    async def finalize_cutover(self, *, document_id, job_id, consumer, event_id, data):
        self.doc.status = DocumentStatus.READY.value
        self.doc.index_version = data.target_index_version
        self.active_version = data.target_index_version  # cutover
        self.job.status = JobStatus.COMPLETED.value
        self.outbox.append(data.completed_event)
        self.inbox.add((consumer, event_id))

    async def mark_failed(
        self, *, document_id, job_id, set_document_failed, consumer, event_id, **_
    ):
        if set_document_failed:
            self.doc.status = DocumentStatus.READY.value  # restore old; active unchanged
            self.inbox.add((consumer, event_id))
        else:
            self.job.status = JobStatus.RETRY_WAIT.value

    async def record_retry(self, job_id, *, attempt, available_in):
        self.job.status = JobStatus.RETRY_WAIT.value
