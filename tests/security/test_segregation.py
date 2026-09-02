"""Domain segregation and identity separation (spec section 16; E2E scenarios 9, 10)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.session.auth import TokenAuthenticator, hash_token
from app.session.identity import Identity
from app.shared.errors import ForbiddenError, UnauthorizedError
from app.testing.harness import make_settings


# --- scenario 10: a deployment cannot be pointed at another domain's stores -----------
@pytest.mark.parametrize(
    "override",
    [
        {"s3_bucket": "warehouse-documents"},
        {"qdrant_collection": "finance-knowledge"},
        {"queue_namespace": "legal"},
    ],
)
def test_isolation_guard_rejects_foreign_stores(override) -> None:
    with pytest.raises(ValidationError):
        make_settings(**override)


# --- scenario 9: chat and ops identities never cross ----------------------------------
def _auth() -> TokenAuthenticator:
    return TokenAuthenticator(chat_hashes=hash_token("chat"), ops_hashes=hash_token("ops"))


def test_chat_token_denied_on_ops_surface() -> None:
    auth = _auth()
    assert auth.authenticate_identity("Bearer chat", Identity.CHAT).identity is Identity.CHAT
    with pytest.raises(ForbiddenError):
        auth.authenticate_identity("Bearer chat", Identity.OPS)


def test_ops_token_denied_on_chat_surface() -> None:
    auth = _auth()
    assert auth.authenticate_identity("Bearer ops", Identity.OPS).identity is Identity.OPS
    with pytest.raises(ForbiddenError):
        auth.authenticate_identity("Bearer ops", Identity.CHAT)


def test_unknown_and_missing_tokens_rejected() -> None:
    auth = _auth()
    with pytest.raises(UnauthorizedError):
        auth.authenticate("Bearer nope")
    with pytest.raises(UnauthorizedError):
        auth.authenticate(None)


def test_chat_and_ops_hashes_must_be_distinct() -> None:
    with pytest.raises(ValueError):
        TokenAuthenticator(chat_hashes=hash_token("same"), ops_hashes=hash_token("same"))


def test_server_requires_both_token_hashes() -> None:
    settings = make_settings(chat_service_token_hash="", ops_service_token_hash="")
    with pytest.raises(ValueError):
        settings.require_server_tokens()
