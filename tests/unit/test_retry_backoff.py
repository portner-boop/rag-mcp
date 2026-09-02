"""Retry backoff schedule (spec section 10)."""

from __future__ import annotations

from app.queueing.consumer import BaseConsumer


class _Consumer(BaseConsumer):
    consumer_name = "test"


def _consumer(jitter: float = 0.0) -> _Consumer:
    return _Consumer(
        store=None,
        domain_id="hr",
        max_attempts=5,
        retry_schedule=(0, 1, 3, 10, 30),
        retry_jitter=jitter,
    )


def test_backoff_follows_schedule() -> None:
    c = _consumer(jitter=0.0)
    assert [c._backoff(a) for a in range(1, 5)] == [1, 3, 10, 30]


def test_backoff_caps_at_last_bucket() -> None:
    c = _consumer(jitter=0.0)
    assert c._backoff(4) == 30
    assert c._backoff(9) == 30  # attempts beyond the schedule stay at the last value


def test_backoff_jitter_is_bounded_and_deterministic() -> None:
    c = _consumer(jitter=1.0)
    for attempt in range(1, 6):
        base = c._retry_schedule[min(attempt, 4)]
        delay = c._backoff(attempt)
        assert base <= delay <= base + 1.0
        assert delay == c._backoff(attempt)  # deterministic (no RNG)
