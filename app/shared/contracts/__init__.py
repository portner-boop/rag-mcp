"""Frozen interservice contracts: MCP tool I/O, queue envelopes and embedding API.

These Pydantic models are the single source of truth for boundary payloads. Producers
and consumers (server + worker) import them; nothing here depends on infrastructure.
Unknown input fields are rejected (spec section 7).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

SCHEMA_VERSION = "1.3.0"


class StrictModel(BaseModel):
    """Base model that rejects unknown fields on input (spec section 7)."""

    model_config = ConfigDict(extra="forbid")
