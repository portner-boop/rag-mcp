from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PipelineResult:
    status: str
    chunk_count: int = 0
    error_code: str | None = None
    error_message: str | None = None
    retryable: bool = False
    stage: str | None = None
