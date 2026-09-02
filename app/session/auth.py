from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass

from app.session.identity import Identity, Scope, scopes_for
from app.shared.errors import ForbiddenError, UnauthorizedError


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Principal:
    identity: Identity
    scopes: frozenset[Scope]

    def require(self, scope: Scope) -> None:
        if scope not in self.scopes:
            raise ForbiddenError(
                "Token is not permitted to call this operation",
                details={"required_scope": scope.value},
            )


class TokenAuthenticator:
    def __init__(self, *, chat_hashes: str, ops_hashes: str) -> None:
        self._chat = self._parse(chat_hashes)
        self._ops = self._parse(ops_hashes)
        if not self._chat or not self._ops:
            raise ValueError("Both chat and ops token hashes are required")
        if self._chat & self._ops:
            raise ValueError("Chat and ops token hashes must not overlap (invariant 2)")

    @staticmethod
    def _parse(raw: str) -> set[str]:
        return {h.strip().lower() for h in raw.split(",") if h.strip()}

    def _match(self, candidate: str, allowed: set[str]) -> bool:
        matched = False
        for allowed_hash in allowed:
            if hmac.compare_digest(candidate, allowed_hash):
                matched = True
        return matched

    def authenticate(self, bearer: str | None) -> Principal:
        if not bearer:
            raise UnauthorizedError("Missing bearer token")
        token = bearer[7:].strip() if bearer.lower().startswith("bearer ") else bearer.strip()
        if not token:
            raise UnauthorizedError("Empty bearer token")
        candidate = hash_token(token)
        if self._match(candidate, self._chat):
            return Principal(Identity.CHAT, scopes_for(Identity.CHAT))
        if self._match(candidate, self._ops):
            return Principal(Identity.OPS, scopes_for(Identity.OPS))
        raise UnauthorizedError("Invalid or revoked service token")

    def authenticate_identity(self, bearer: str | None, expected: Identity) -> Principal:
        principal = self.authenticate(bearer)
        if principal.identity is not expected:
            raise ForbiddenError(
                "This token identity may not access this surface",
                details={"expected_identity": expected.value},
            )
        return principal
