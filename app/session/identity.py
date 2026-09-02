from __future__ import annotations

from enum import Enum


class Identity(str, Enum):
    CHAT = "chat"
    OPS = "ops"


class Scope(str, Enum):
    KNOWLEDGE_SEARCH = "knowledge:search"
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
