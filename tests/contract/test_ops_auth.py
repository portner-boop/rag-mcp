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
    resp = client.post(
        "/internal/documents/delete", json={}, headers={"Authorization": f"Bearer {OPS_TOKEN}"}
    )
    assert resp.status_code == 422
