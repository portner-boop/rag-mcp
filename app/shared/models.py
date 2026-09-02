"""Shared response models (bpmn-mcp Envelope convention).

`search_knowledge` returns its spec-frozen output shape directly (spec 7.1), so it does
NOT wrap success in this Envelope. The Envelope + ok()/fail() helpers are used for
non-contract responses (internal acknowledgements, health payloads) that benefit from a
uniform shape.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Envelope(BaseModel):
    ok: bool
    result: Any = Field(default_factory=dict)
    warnings: list[Any] = Field(default_factory=list)
    errors: list[Any] = Field(default_factory=list)
    hints: list[Any] = Field(default_factory=list)
    code: str | None = None
