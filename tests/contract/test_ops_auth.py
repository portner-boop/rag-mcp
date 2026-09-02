"""Operational interface auth contract (spec sections 6, 7; E2E scenario 9).

Denials use ``/internal/documents/find`` whose body is fully optional, so an empty ``{}``
is a valid body and the response is decided purely by the auth dependency. The chat token
must never reach an operational command.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.operational.api import build_ops_app
from app.testing.harness import CHAT_TOKEN, OPS_TOKEN, make_container

client = TestClient(build_ops_app(make_container()))

FIND = "/internal/documents/find"


def test_missing_token_is_unauthorized() -> None:
    resp = client.post(FIND, json={})
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "UNAUTHORIZED"


def test_chat_token_is_forbidden_on_ops() -> None:
    resp = client.post(FIND, json={}, headers={"Authorization": f"Bearer {CHAT_TOKEN}"})
    assert resp.status_code == 403
    assert resp.json()["error"]["code"] == "FORBIDDEN"


def test_unknown_token_is_unauthorized() -> None:
    resp = client.post(FIND, json={}, headers={"Authorization": "Bearer nope"})
    assert resp.status_code == 401


def test_ops_token_clears_auth_then_body_validates() -> None:
    # A valid ops token clears auth; an empty body for a required-field route then fails
    # request validation (422), proving the token was accepted (not 401/403).
    resp = client.post(
        "/internal/documents/delete", json={}, headers={"Authorization": f"Bearer {OPS_TOKEN}"}
    )
    assert resp.status_code == 422
