"""Transactional-outbox relay (spec section 8.2, 10).

Polls ``outbox_events`` for PENDING rows, publishes each with publisher confirms and
marks it PUBLISHED in the same transaction it was locked in (SELECT ... FOR UPDATE SKIP
LOCKED). A crash between publish and mark re-publishes the row; consumers dedupe by
``event_id`` (at-least-once delivery, exactly-once business effect).
"""

from __future__ import annotations

import asyncio

import structlog

from app.storage.postgres.engine import Database
from app.storage.postgres.repositories import OutboxRepository
from app.storage.queue.publisher import Publisher

log = structlog.get_logger("outbox_relay")


class OutboxRelay:
    def __init__(
        self,
        *,
        database: Database,
        publisher: Publisher,
        poll_interval: float = 1.0,
        batch_size: int = 50,
        failure_backoff_seconds: int = 5,
    ) -> None:
        self._db = database
        self._publisher = publisher
        self._poll_interval = poll_interval
        self._batch_size = batch_size
        self._backoff = failure_backoff_seconds
        self._stop = asyncio.Event()

    def stop(self) -> None:
        self._stop.set()

    async def run(self) -> None:
        log.info("outbox_relay.start")
        while not self._stop.is_set():
            try:
                published = await self._drain_once()
            except Exception:  # noqa: BLE001
                log.exception("outbox_relay.error")
                published = 0
            if published == 0:
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=self._poll_interval)
                except TimeoutError:
                    pass
        log.info("outbox_relay.stop")

    async def _drain_once(self) -> int:
        async with self._db.session() as session:
            repo = OutboxRepository(session)
            rows = await repo.fetch_pending(limit=self._batch_size)
            published = 0
            for row in rows:
                try:
                    await self._publisher.publish(
                        routing_key=row.routing_key,
                        body=row.payload,
                        message_id=row.event_id,
                        trace_id=row.payload.get("trace_id"),
                    )
                    await repo.mark_published(row)
                    published += 1
                except Exception:  # noqa: BLE001
                    log.warning("outbox_relay.publish_failed", event_id=row.event_id)
                    await repo.mark_failed(row, backoff_seconds=self._backoff)
            return published
