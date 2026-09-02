from __future__ import annotations

import aio_pika
from aio_pika.abc import AbstractRobustChannel, AbstractRobustConnection


async def connect(url: str) -> AbstractRobustConnection:
    return await aio_pika.connect_robust(url)


async def open_channel(
    connection: AbstractRobustConnection, *, prefetch: int | None = None
) -> AbstractRobustChannel:
    channel = await connection.channel(publisher_confirms=True)
    if prefetch is not None:
        await channel.set_qos(prefetch_count=prefetch)
    return channel  # type: ignore[return-value]
