"""Per-domain RabbitMQ topology (spec section 10).

    exchange: domain.<domain>.commands   (direct)
    queues:   domain.<domain>.ingestion | deletion | reindex   (quorum, dead-lettered)
    dlx:      domain.<domain>.dlx
    dlq:      domain.<domain>.dlq

Queues are durable quorum queues; messages are persistent; the exchange is declared
durable so the topology survives a broker restart.
"""

from __future__ import annotations

from dataclasses import dataclass

import aio_pika
from aio_pika.abc import AbstractExchange, AbstractQueue, AbstractRobustChannel

QUEUE_KINDS = ("ingestion", "deletion", "reindex")


@dataclass
class Topology:
    exchange: AbstractExchange
    dlx: AbstractExchange
    queues: dict[str, AbstractQueue]
    dlq: AbstractQueue
    events: AbstractQueue


async def declare_topology(
    channel: AbstractRobustChannel,
    *,
    exchange_name: str,
    dlx_name: str,
    dlq_name: str,
    queue_name: callable,  # type: ignore[valid-type]
    routing_key: callable,  # type: ignore[valid-type]
) -> Topology:
    exchange = await channel.declare_exchange(
        exchange_name, aio_pika.ExchangeType.DIRECT, durable=True
    )
    dlx = await channel.declare_exchange(dlx_name, aio_pika.ExchangeType.DIRECT, durable=True)

    dlq = await channel.declare_queue(dlq_name, durable=True, arguments={"x-queue-type": "quorum"})
    await dlq.bind(dlx, routing_key="#")

    queues: dict[str, AbstractQueue] = {}
    for kind in QUEUE_KINDS:
        q = await channel.declare_queue(
            queue_name(kind),
            durable=True,
            arguments={
                "x-queue-type": "quorum",
                "x-dead-letter-exchange": dlx_name,
                "x-dead-letter-routing-key": kind,
            },
        )
        await q.bind(exchange, routing_key=routing_key(kind))
        queues[kind] = q

    # Lifecycle notifications (ingestion/deletion/reindex completed|failed). Durable,
    # no dead-lettering: it is an audit/notification stream, consumed by observability.
    events = await channel.declare_queue(
        queue_name("events"), durable=True, arguments={"x-queue-type": "quorum"}
    )
    await events.bind(exchange, routing_key=routing_key("events"))

    return Topology(exchange=exchange, dlx=dlx, queues=queues, dlq=dlq, events=events)
