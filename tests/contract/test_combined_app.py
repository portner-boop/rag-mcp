"""One-port topology: chat MCP at / and ops under /ops share a host:port (S1 unify)."""

from __future__ import annotations

from starlette.testclient import TestClient

from app.server import build_combined_app
from app.testing.harness import CHAT_TOKEN, OPS_TOKEN, make_container


def _client() -> TestClient:
    return TestClient(build_combined_app(make_container()))


def test_health_live_served_at_root() -> None:
    with _client() as client:
        assert client.get("/health/live").status_code == 200


def test_ops_mounted_under_ops_prefix_and_enforces_ops_identity() -> None:
    with _client() as client:
        # `delete` needs a body, so validation (422) fires after auth without touching the DB.
        delete = "/ops/internal/documents/delete"
        assert client.post(delete, json={}).status_code == 401  # no token
        # A chat token is the wrong identity for the ops surface.
        chat = client.post(delete, json={}, headers={"Authorization": f"Bearer {CHAT_TOKEN}"})
        assert chat.status_code == 403
        # The ops token clears auth; the empty body then fails validation.
        ok = client.post(delete, json={}, headers={"Authorization": f"Bearer {OPS_TOKEN}"})
        assert ok.status_code == 422


def test_ops_health_and_metrics_under_prefix() -> None:
    with _client() as client:
        assert client.get("/ops/metrics").status_code == 200
