"""Deterministic fake embeddings shared by the fake API and in-process fakes.

Vectors depend only on the input text, so ingestion and query embeddings match and the
same text always yields the same vector (needed for reproducible smoke checks).
"""

from __future__ import annotations

import hashlib
import math

from app.shared.contracts.embedding import SparseVector


def _seed(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def dense_vector(text: str, dimension: int) -> list[float]:
    seed = _seed(text) or 1
    values: list[float] = []
    state = seed
    for _ in range(dimension):
        # Deterministic LCG (glibc constants).
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        values.append((state / 0x7FFFFFFF) - 0.5)
    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return [v / norm for v in values]


def sparse_vector(text: str, *, vocab: int = 4096, top_k: int = 8) -> SparseVector:
    tokens = [t for t in text.lower().split() if t]
    weights: dict[int, float] = {}
    for token in tokens:
        idx = _seed(token) % vocab
        weights[idx] = weights.get(idx, 0.0) + 1.0
    ordered = sorted(weights.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
    ordered.sort(key=lambda kv: kv[0])
    indices = [i for i, _ in ordered]
    values = [v for _, v in ordered]
    return SparseVector(indices=indices, values=values)


def rerank_score(query: str, text: str) -> float:
    q = set(query.lower().split())
    d = set(text.lower().split())
    if not q or not d:
        return 0.0
    return len(q & d) / len(q | d)
