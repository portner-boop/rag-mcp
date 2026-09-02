"""Consumer dispatch: manual ack, handler-twice idempotency, retry, DLQ, cross-domain.

Drives the real IngestionConsumer (BaseConsumer subclass) with a FakeMessage so ack/
nack/reject and the retry/DLQ branches are exercised without a broker (spec 10, 11).
"""

from __future__ import annotations

import json

from app.ingestion.consumer import IngestionConsumer
from app.shared.enums import DocumentStatus
from app.testing.harness import (
    FakeMessage,
    build_ingestion_pipeline,
    ingestion_setup,
    make_ingestion_event,
)

POLICY = "# Doc\n\n" + ("leave transfer policy words here " * 20)


def _consumer(setup, *, max_attempts=5):
    return IngestionConsumer(
        pipeline_factory=lambda owner: build_ingestion_pipeline(setup, owner=owner),
        store=setup.store,
        domain_id="hr",
        owner="w1",
        max_attempts=max_attempts,
        retry_schedule=(0, 0, 0, 0, 0),
        retry_jitter=0.0,
    )


async def test_handler_twice_is_idempotent_and_acks() -> None:
    setup = ingestion_setup(text=POLICY)
    consumer = _consumer(setup)

    msg1 = FakeMessage.from_event(setup.event)
    await consumer.handle(msg1)
    assert msg1.acked
    count1 = await setup.vectors.count_for_document(setup.document_id, index_version=1)
    assert setup.store.docs[setup.document_id].state.status == DocumentStatus.READY.value

    # Same delivery again -> duplicate, still acked, no double effect.
    msg2 = FakeMessage.from_event(setup.event)
    await consumer.handle(msg2)
    assert msg2.acked
    assert await setup.vectors.count_for_document(setup.document_id, index_version=1) == count1


async def test_cross_domain_payload_is_rejected() -> None:
    setup = ingestion_setup(text=POLICY)
    consumer = _consumer(setup)
    foreign = make_ingestion_event(
        setup.document_id, setup.job_id, setup.original_key, domain="warehouse"
    )
    msg = FakeMessage.from_event(foreign)
    await consumer.handle(msg)
    assert msg.rejected and msg.requeue is False  # dead-lettered, not requeued
    # No processing happened.
    assert setup.store.docs[setup.document_id].state.status == DocumentStatus.QUEUED.value


async def test_malformed_body_is_dead_lettered() -> None:
    setup = ingestion_setup(text=POLICY)
    consumer = _consumer(setup)
    msg = FakeMessage(body=b"{not json")
    await consumer.handle(msg)
    assert msg.rejected and msg.requeue is False


async def test_retryable_failure_nacks_for_redelivery() -> None:
    setup = ingestion_setup(text=POLICY)

    async def boom(_texts):
        from app.shared.errors import ErrorCode, UpstreamError

        raise UpstreamError("down", code=ErrorCode.EMBEDDING_TIMEOUT, retryable=True)

    setup.embedding.dense = boom  # type: ignore[method-assign]
    consumer = _consumer(setup, max_attempts=5)
    msg = FakeMessage.from_event(setup.event)
    await consumer.handle(msg)
    assert msg.nacked and msg.requeue is True  # retry -> requeue
    assert not msg.acked


async def test_exhausted_retries_dead_letter_and_set_failed() -> None:
    setup = ingestion_setup(text=POLICY)

    async def boom(_texts):
        from app.shared.errors import ErrorCode, UpstreamError

        raise UpstreamError("down", code=ErrorCode.EMBEDDING_TIMEOUT, retryable=True)

    setup.embedding.dense = boom  # type: ignore[method-assign]
    consumer = _consumer(setup, max_attempts=1)  # attempt 0 -> next 1 == max -> terminal
    msg = FakeMessage.from_event(setup.event)
    await consumer.handle(msg)
    assert msg.rejected and msg.requeue is False  # dead-lettered
    assert setup.store.docs[setup.document_id].state.status == DocumentStatus.FAILED.value
    body = json.loads(FakeMessage.from_event(setup.event).body)
    assert body["event_type"] == "DocumentIngestionRequested"  # sanity on the harness
