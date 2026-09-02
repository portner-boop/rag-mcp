"""Dead-letter queue inspect + redrive under the operational identity (spec 10; D04).

Inspect peeks messages without consuming them (get then requeue). Redrive republishes
dead-lettered messages to the domain command exchange on their original routing key; the
original ``event_id`` is preserved so business idempotency still holds.
"""

from __future__ import annotations

import json

import structlog
from aio_pika.abc import AbstractExchange, AbstractRobustChannel

log = structlog.get_logger("dlq")


class DlqService:
    def __init__(
        self,
        *,
        channel: AbstractRobustChannel,
        exchange: AbstractExchange,
        dlq_name: str,
        routing_key,  # settings.routing_key
    ) -> None:
        self._channel = channel
        self._exchange = exchange
        self._dlq_name = dlq_name
        self._routing_key = routing_key

    async def inspect(self, limit: int = 20) -> list[dict]:
        queue = await self._channel.get_queue(self._dlq_name, ensure=False)
        out: list[dict] = []
        seen = []
        for _ in range(limit):
            message = await queue.get(no_ack=False, fail=False)
            if message is None:
                break
            seen.append(message)
            try:
                body = json.loads(message.body.decode("utf-8"))
            except Exception:  # noqa: BLE001
                body = {"_unparseable": True}
            out.append(
                {
                    "event_id": body.get("event_id"),
                    "event_type": body.get("event_type"),
                    "document_id": body.get("document_id"),
                    "routing_key": message.routing_key,
                }
            )
        # Requeue everything we peeked so inspect is non-destructive.
        for message in seen:
            await message.nack(requeue=True)
        return out

    async def redrive(self, limit: int = 20) -> int:
        queue = await self._channel.get_queue(self._dlq_name, ensure=False)
        moved = 0
        for _ in range(limit):
            message = await queue.get(no_ack=False, fail=False)
            if message is None:
                break
            # DLQ routing key is the queue "kind" (ingestion|deletion|reindex).
            kind = message.routing_key
            await self._exchange.publish(message, routing_key=self._routing_key(kind))
            await message.ack()
            moved += 1
        if moved:
            log.info("dlq.redriven", count=moved)
        return moved
