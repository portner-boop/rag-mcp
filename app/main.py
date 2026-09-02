"""Single entrypoint with subcommands (the worker is a mode of the app, not a package).

rag-mcp                # or `rag-mcp serve` — chat MCP + operational API
rag-mcp worker         # ingestion queue consumer
rag-mcp provision      # one-shot: seed active index config + Qdrant collection
"""

from __future__ import annotations

import asyncio
import sys

from app.config import get_settings
from app.logging import setup_logging


def _configure(service: str) -> None:
    settings = get_settings()
    setup_logging(
        settings.log_level,
        service=service,
        domain=settings.domain_id,
        environment=settings.environment,
    )


async def _provision() -> None:
    from app.container import build_container
    from app.provisioning import provision

    container = build_container()
    await provision(container.database, container.qdrant, container.settings)
    await container.shutdown()


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "serve"

    if mode == "worker":
        _configure("domain-mcp-worker")
        from app import worker

        asyncio.run(worker.run(get_settings()))
    elif mode == "provision":
        _configure("domain-mcp-provision")
        asyncio.run(_provision())
    elif mode in ("serve", ""):
        _configure("domain-mcp-server")
        from app import server

        asyncio.run(server.serve(get_settings()))
    else:
        raise SystemExit(f"Unknown mode: {mode!r} (use serve | worker | provision)")


if __name__ == "__main__":
    main()
