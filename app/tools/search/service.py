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
