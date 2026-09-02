from __future__ import annotations

import asyncio
import json

import structlog
from aio_pika.abc import AbstractIncomingMessage

from app.observability import metrics
from app.shared.contracts.queue import EventEnvelope, parse_event
from app.shared.enums import EventType
from app.shared.trace import parse_traceparent, set_current
from app.worker_support.result import PipelineResult

log = structlog.get_logger("consumer")


class BaseConsumer:
    consumer_name: str
    expected_event_type: EventType

    def __init__(
        self,
        *,
        store,
        domain_id: str,
        max_attempts: int,
        retry_schedule: tuple[int, ...],
        retry_jitter: float,
    ) -> None:
        self._store = store
        self._domain = domain_id
        self._max_attempts = max_attempts
        self._retry_schedule = retry_schedule
        self._jitter = retry_jitter

    async def run_pipeline(self, event: EventEnvelope) -> PipelineResult:
        raise NotImplementedError

    def build_failed_event(self, event: EventEnvelope, result: PipelineResult) -> dict:
        raise NotImplementedError

    def _metric_success(self, result: PipelineResult) -> None:
        metrics.worker_jobs_total.labels(consumer=self.consumer_name, result=result.status).inc()

    def _metric_failed(self) -> None:
        metrics.worker_jobs_total.labels(consumer=self.consumer_name, result="failed").inc()

    def _metric_retry(self) -> None:
        metrics.worker_retries_total.labels(consumer=self.consumer_name).inc()

    def _metric_dlq(self) -> None:
        metrics.worker_dlq_total.labels(consumer=self.consumer_name).inc()

    async def handle(self, message: AbstractIncomingMessage) -> None:
        headers = message.headers or {}
        set_current(parse_traceparent(headers.get("traceparent"), headers.get("trace_id")))
        try:
            event = parse_event(json.loads(message.body.decode("utf-8")))
        except Exception:  # noqa: BLE001
            log.error("consumer.deserialize_failed", consumer=self.consumer_name)
            self._metric_dlq()
            await message.reject(requeue=False)
            return

        if event.event_type != self.expected_event_type:
            log.error("consumer.wrong_event_type", consumer=self.consumer_name)
            await message.reject(requeue=False)
            return
        if event.domain != self._domain:
            log.critical(
                "consumer.cross_domain_rejected",
                consumer=self.consumer_name,
                event_domain=event.domain,
                expected=self._domain,
            )
            self._metric_dlq()
            await message.reject(requeue=False)
            return

        result = await self.run_pipeline(event)
        if result.status in ("completed", "duplicate"):
            self._metric_success(result)
            await message.ack()
            return
        await self._on_failure(message, event, result)

    async def _on_failure(
        self, message: AbstractIncomingMessage, event: EventEnvelope, result: PipelineResult
    ) -> None:
        self._metric_failed()
        next_attempt = event.attempt + 1
        if result.retryable and next_attempt < self._max_attempts:
            delay = self._backoff(next_attempt)
            await self._store.record_retry(event.job_id, attempt=next_attempt, available_in=delay)
            self._metric_retry()
            log.info(
                "consumer.retry", consumer=self.consumer_name, attempt=next_attempt, delay=delay
            )
            await asyncio.sleep(delay)
            await message.nack(requeue=True)
            return

        failed_event = self.build_failed_event(event, result)
        failed_event["_dead_letter"] = result.retryable
        await self._store.mark_failed(
            document_id=event.document_id,
            job_id=event.job_id,
            stage=result.stage or "UNKNOWN",
            error_code=result.error_code or "INTERNAL",
            error_message=result.error_message or "",
            attempt=next_attempt,
            set_document_failed=True,
            failed_event=failed_event,
            consumer=self.consumer_name,
            event_id=event.event_id,
        )
        self._metric_dlq()
        log.warning("consumer.dead_letter", consumer=self.consumer_name, code=result.error_code)
        await message.reject(requeue=False)

    def _backoff(self, attempt: int) -> float:
        idx = min(attempt, len(self._retry_schedule) - 1)
        base = self._retry_schedule[idx]
        return base + (self._jitter * ((attempt % 3) / 2.0))
