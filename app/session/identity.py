"""Service identities and scopes (spec sections 2 and 6).

The chat orchestrator authenticates with a token whose only scope is
``knowledge:search``. The internal operational control plane authenticates with a
separate token carrying corpus-maintenance scopes. The two tokens are distinct and
cannot substitute for each other (invariant 2).
"""

from __future__ import annotations

from enum import Enum


class Identity(str, Enum):
    CHAT = "chat"
    OPS = "ops"


class Scope(str, Enum):
    # Chat-facing scope.
    KNOWLEDGE_SEARCH = "knowledge:search"
    # Operational corpus scopes.
    CORPUS_READ = "corpus:read"
    CORPUS_WRITE = "corpus:write"
    CORPUS_DELETE = "corpus:delete"
    CORPUS_REINDEX = "corpus:reindex"


CHAT_SCOPES: frozenset[Scope] = frozenset({Scope.KNOWLEDGE_SEARCH})

OPS_SCOPES: frozenset[Scope] = frozenset(
    {Scope.CORPUS_READ, Scope.CORPUS_WRITE, Scope.CORPUS_DELETE, Scope.CORPUS_REINDEX}
)


def scopes_for(identity: Identity) -> frozenset[Scope]:
    return CHAT_SCOPES if identity is Identity.CHAT else OPS_SCOPES
