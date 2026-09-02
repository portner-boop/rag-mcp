"""search_knowledge orchestration (tool layer).

Builds the validated input contract, calls the shared retrieval engine
(`app.search.service`) via the process container, and returns the spec-frozen output
shape (spec 7.1) as a dict. This success shape is NOT wrapped in the generic Envelope —
the orchestrator expects the exact search_knowledge fields.
"""

from __future__ import annotations

from app.shared.contracts.mcp import SearchFilters, SearchKnowledgeInput
from app.tools import _common


async def run(
    *,
    query: str,
    limit: int,
    max_candidates: int,
    document_ids: list[str] | None,
    filters: SearchFilters | None,
) -> dict:
    payload = SearchKnowledgeInput(
        query=query,
        limit=limit,
        max_candidates=max_candidates,
        document_ids=document_ids,
        filters=filters or SearchFilters(),
    )
    result = await _common.container().search_service.search_knowledge(payload)
    return result.model_dump(mode="json")
