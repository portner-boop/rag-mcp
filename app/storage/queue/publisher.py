"""Message publisher with publisher confirms and trace propagation (spec section 10, 17)."""

from __future__ import annotations

import json

import aio_pika
from aio_pika.abc import AbstractExchange

from app.shared.errors import UpstreamError


class Publisher:
    def __init__(self, exchange: AbstractExchange) -> None:
        self._exchange = exchange

    async def publish(
        self,
        *,
        routing_key: str,
        body: dict,
        message_id: str,
        trace_id: str | None = None,
    ) -> None:
        headers: dict[str, str] = {}
        if trace_id:
            headers["traceparent"] = f"00-{trace_id}-{message_id.replace('-', '')[:16]}-01"
            headers["trace_id"] = trace_id
        message = aio_pika.Message(
            body=json.dumps(body, separators=(",", ":")).encode("utf-8"),
            content_type="application/json",
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            message_id=message_id,
            headers=headers,
        )
        try:
            await self._exchange.publish(message, routing_key=routing_key)
        except Exception as exc:  # noqa: BLE001
            raise UpstreamError("Queue publish failed") from exc
