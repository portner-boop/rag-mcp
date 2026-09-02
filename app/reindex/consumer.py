from __future__ import annotations

from app.queueing.consumer import BaseConsumer
from app.reindex.pipeline import ReindexPipeline
from app.shared.contracts.queue import (
    DocumentReindexFailed,
    DocumentReindexRequested,
    EventEnvelope,
)
from app.shared.enums import EventType
from app.shared.ids import new_uuid
from app.shared.time import to_rfc3339, utcnow
from app.worker_support.result import PipelineResult

CONSUMER_NAME = "reindex-worker"


class ReindexConsumer(BaseConsumer):
    consumer_name = CONSUMER_NAME
    expected_event_type = EventType.DOCUMENT_REINDEX_REQUESTED

    def __init__(self, *, pipeline: ReindexPipeline, store, domain_id: str, **kwargs) -> None:
        super().__init__(store=store, domain_id=domain_id, **kwargs)
        self._pipeline = pipeline

    async def run_pipeline(self, event: EventEnvelope) -> PipelineResult:
        assert isinstance(event, DocumentReindexRequested)
        return await self._pipeline.run(event)

    def build_failed_event(self, event: EventEnvelope, result: PipelineResult) -> dict:
        return DocumentReindexFailed(
            event_id=new_uuid(),
            occurred_at=to_rfc3339(utcnow()),
            domain=self._domain,
            document_id=event.document_id,
            job_id=event.job_id,
            attempt=event.attempt,
            trace_id=event.trace_id,
            target_index_version=event.target_index_version,
            error_code=result.error_code or "INTERNAL",
            retryable=result.retryable,
        ).model_dump(mode="json")
