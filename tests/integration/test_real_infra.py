"""Real-infrastructure integration tests (spec section 18).

These require live PostgreSQL, RabbitMQ, MinIO and Qdrant (Testcontainers) plus the fake
Embedding API, so they are skipped unless RUN_INTEGRATION=1. The fake-backed E2E suite
(``test_e2e_scenarios.py``) covers the same behaviours without a broker/daemon; this file
pins the remaining real-infra scope required by the release gate.

Enable locally with:

    RUN_INTEGRATION=1 uv run pytest tests/integration/test_real_infra.py
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION") != "1",
    reason="requires Docker infra; set RUN_INTEGRATION=1 to run",
)


@pytest.mark.skip(reason="D05 real-infra harness: Alembic upgrade on empty PostgreSQL + fixtures")
def test_migrations_apply_on_empty_postgres() -> None:
    """`alembic upgrade head` succeeds on an empty PostgreSQL 17 container."""


@pytest.mark.skip(reason="D05 real-infra harness: ingest -> READY -> search with real Qdrant")
def test_ingestion_to_search_roundtrip() -> None:
    """prepare_upload -> PUT to MinIO -> start_ingestion -> worker -> READY -> search_knowledge."""


@pytest.mark.skip(reason="D05 real-infra harness: kill worker mid-stage, resume after lease expiry")
def test_worker_crash_and_lease_recovery() -> None:
    """A worker killed between stages is recovered after lease expiry; no false READY."""


@pytest.mark.skip(reason="D05 real-infra harness: RabbitMQ redelivery -> exactly-once effect")
def test_at_least_once_delivery_exactly_once_effect() -> None:
    """A redelivered DocumentIngestionRequested produces exactly one corpus effect."""


@pytest.mark.skip(reason="D05 real-infra harness: HR credentials denied on warehouse bucket/Qdrant")
def test_cross_domain_storage_denied() -> None:
    """HR S3/Qdrant credentials cannot list/read another domain's stores."""


@pytest.mark.skip(reason="D05 load/soak: search + queue + workers under sustained load meet SLO")
def test_load_and_soak() -> None:
    """Search p95 and ingestion throughput meet the agreed SLO under sustained load."""
