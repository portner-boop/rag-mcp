"""Shared tool-layer helpers (bpmn-mcp `tools/_common.py`).

The process-wide container (built at server startup) is resolved lazily here so tool
bodies stay one-line delegates. Trace context is bound by the ASGI auth middleware ahead
of the tool call, so tools need no `Context` for correlation.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from app.container import Container, get_container

OptionalDocumentIds = Annotated[
    list[str] | None,
    Field(
        default=None,
        description=(
            "Restrict the search to these document ids in the current domain. Every id must "
            "exist in this deployment. Omit to search the whole corpus."
        ),
    ),
]


def container() -> Container:
    return get_container()
