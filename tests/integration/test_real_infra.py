from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION") != "1",
    reason="requires Docker infra; set RUN_INTEGRATION=1 to run",
)


@pytest.mark.skip(reason="D05 real-infra harness: Alembic upgrade on empty PostgreSQL + fixtures")
def test_migrations_apply_on_empty_postgres() -> None:
    pass


@pytest.mark.skip(reason="D05 real-infra harness: ingest -> READY -> search with real Qdrant")
def test_ingestion_to_search_roundtrip() -> None:
    pass


@pytest.mark.skip(reason="D05 real-infra harness: kill worker mid-stage, resume after lease expiry")
def test_worker_crash_and_lease_recovery() -> None:
    pass


@pytest.mark.skip(reason="D05 real-infra harness: RabbitMQ redelivery -> exactly-once effect")
def test_at_least_once_delivery_exactly_once_effect() -> None:
    pass


@pytest.mark.skip(reason="D05 real-infra harness: HR credentials denied on warehouse bucket/Qdrant")
def test_cross_domain_storage_denied() -> None:
    pass


@pytest.mark.skip(reason="D05 load/soak: search + queue + workers under sustained load meet SLO")
def test_load_and_soak() -> None:
    pass
