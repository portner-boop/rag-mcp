"""Robust RabbitMQ connection helpers shared by publisher and worker consumer."""

from __future__ import annotations

import aio_pika
from aio_pika.abc import AbstractRobustChannel, AbstractRobustConnection


async def connect(url: str) -> AbstractRobustConnection:
    return await aio_pika.connect_robust(url)


async def open_channel(
    connection: AbstractRobustConnection, *, prefetch: int | None = None
) -> AbstractRobustChannel:
    # publisher_confirms defaults to True in aio-pika: publish awaits broker ack.
    channel = await connection.channel(publisher_confirms=True)
    if prefetch is not None:
        await channel.set_qos(prefetch_count=prefetch)
    return channel  # type: ignore[return-value]
