from __future__ import annotations

from pydantic import BaseModel, ConfigDict

SCHEMA_VERSION = "1.3.0"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
