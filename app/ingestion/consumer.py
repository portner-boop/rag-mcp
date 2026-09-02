from __future__ import annotations

from app.observability import metrics
from app.queueing.consumer import BaseConsumer
from app.shared.contracts.queue import (
    DocumentIngestionFailed,
    DocumentIngestionRequested,
    EventEnvelope,
)
from app.shared.enums import EventType
from app.shared.ids import new_uuid
from app.shared.time import to_rfc3339, utcnow
from app.worker_support.result import PipelineResult

CONSUMER_NAME = "ingestion-worker"


class IngestionConsumer(BaseConsumer):
    consumer_name = CONSUMER_NAME
    expected_event_type = EventType.DOCUMENT_INGESTION_REQUESTED

    def __init__(self, *, pipeline_factory, store, domain_id: str, owner: str, **kwargs) -> None:
        kwargs.pop("consumer_name", None)
        super().__init__(store=store, domain_id=domain_id, **kwargs)
        self._pipeline_factory = pipeline_factory
        self._owner = owner

    async def run_pipeline(self, event: EventEnvelope) -> PipelineResult:
        assert isinstance(event, DocumentIngestionRequested)
        with metrics.ingestion_job_duration_seconds.time():
            return await self._pipeline_factory(self._owner).run(event)

    def build_failed_event(self, event: EventEnvelope, result: PipelineResult) -> dict:
        return DocumentIngestionFailed(
            event_id=new_uuid(),
            occurred_at=to_rfc3339(utcnow()),
            domain=self._domain,
            document_id=event.document_id,
            job_id=event.job_id,
            attempt=event.attempt,
            trace_id=event.trace_id,
            stage=result.stage or "UNKNOWN",
            error_code=result.error_code or "INTERNAL",
            retryable=result.retryable,
        ).model_dump(mode="json")

    def _metric_success(self, result: PipelineResult) -> None:
        metrics.ingestion_jobs_total.labels(result=result.status).inc()
        if result.chunk_count:
            metrics.chunks_created_total.inc(result.chunk_count)
            metrics.qdrant_points_upserted_total.inc(result.chunk_count)

    def _metric_failed(self) -> None:
        metrics.ingestion_jobs_failed_total.inc()

    def _metric_retry(self) -> None:
        metrics.ingestion_retries_total.inc()

    def _metric_dlq(self) -> None:
        metrics.ingestion_dlq_total.inc()
