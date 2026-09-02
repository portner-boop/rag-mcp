"""The chat-facing MCP surface.

Creates `public_mcp` and imports each tool package so its `@public_mcp.tool` decorator
fires on import. Only knowledge retrieval is published — ingestion/corpus maintenance is
a separate operational identity/network path (`app.operational`) and is never a tool
here (invariant 2). A tool package never imports another tool package.
"""

from fastmcp import FastMCP

public_mcp = FastMCP(name="domain-knowledge")

from app.tools import search  # noqa: E402,F401
