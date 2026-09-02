"""Shared pipeline result type (ingestion / deletion / reindex).

Neutral home so the deletion and reindex engines do not import from the ingestion
pipeline just to reuse this dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PipelineResult:
    status: str  # "completed" | "duplicate" | "failed"
    chunk_count: int = 0
    error_code: str | None = None
    error_message: str | None = None
    retryable: bool = False
    stage: str | None = None
