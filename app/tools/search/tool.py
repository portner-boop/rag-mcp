from __future__ import annotations

from app.shared.contracts.mcp import SearchFilters
from app.shared.envelope import tool_wrapper
from app.tools import public_mcp
from app.tools._common import OptionalDocumentIds


@public_mcp.tool(
    name="search_knowledge",
    description=(
        "Search this domain's knowledge base and return cited chunks with immutable "
        "citation metadata. Provide a natural-language query; optionally restrict to "
        "specific document ids or a created/version filter. Returns query_id, results and "
        "search_meta. Never returns file URLs."
    ),
)
@tool_wrapper
async def search_knowledge(
    query: str,
    limit: int = 10,
    max_candidates: int = 50,
    document_ids: OptionalDocumentIds = None,
    filters: SearchFilters | None = None,
) -> dict:
    from app.tools.search import service

    return await service.run(
        query=query,
        limit=limit,
        max_candidates=max_candidates,
        document_ids=document_ids,
        filters=filters,
    )
