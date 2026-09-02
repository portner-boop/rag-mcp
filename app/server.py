from __future__ import annotations

import asyncio

import structlog
import uvicorn
from fastmcp import FastMCP
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.config import Settings
from app.container import Container, set_container
from app.operational.api import build_ops_app
from app.provisioning import provision
from app.session.identity import Identity
from app.shared.enums import CapabilityProfile
from app.shared.errors import DomainError
from app.shared.trace import parse_traceparent, set_current
from app.tools import public_mcp

log = structlog.get_logger("server")

_HEALTH_PREFIX = "/health"


class ChatAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, container: Container) -> None:  # noqa: ANN001
        super().__init__(app)
        self._container = container

    async def dispatch(self, request: Request, call_next):  # noqa: ANN001
        ctx = parse_traceparent(
            request.headers.get("traceparent"), request.headers.get("x-request-id")
        )
        set_current(ctx)
        if request.url.path.startswith(_HEALTH_PREFIX):
            return await call_next(request)
        try:
            self._container.authenticator.authenticate_identity(
                request.headers.get("authorization"), Identity.CHAT
            )
        except DomainError as exc:
            return JSONResponse(exc.to_contract(ctx.request_id), status_code=exc.http_status)
        return await call_next(request)


def build_mcp(container: Container) -> FastMCP:
    settings = container.settings
    mcp = FastMCP(
        name=f"domain-knowledge-mcp-{settings.domain_id}",
        instructions=(
            "Knowledge retrieval MCP for a single business domain. "
            "Only the search_knowledge tool is available."
        ),
        version=settings.contract_version,
    )
    mcp.mount(public_mcp)

    @mcp.custom_route(f"{_HEALTH_PREFIX}/live", methods=["GET"])
    async def live(_request: Request) -> Response:  # noqa: ANN202
        return JSONResponse({"status": "live"})

    @mcp.custom_route(f"{_HEALTH_PREFIX}/ready", methods=["GET"])
    async def ready(_request: Request) -> Response:  # noqa: ANN202
        report = await container.chat_readiness()
        body = {
            "status": "ready" if report.ok else "not_ready",
            "domain": settings.domain_id,
            "contract_version": settings.contract_version,
            "capability_profile": CapabilityProfile(settings.capability_profile).value,
            "search_flags": {
                "dense": True,
                "sparse": True,
                "rerank": settings.enable_reranker,
                "fallback_dense_only": settings.allow_dense_only_fallback,
                "fallback_sparse_only": settings.allow_sparse_only_fallback,
            },
            "checks": report.checks,
        }
        return JSONResponse(body, status_code=200 if report.ok else 503)

    return mcp


def build_mcp_app(container: Container):
    mcp = build_mcp(container)
    app = mcp.http_app(path=container.settings.mcp_path, transport="streamable-http")
    app.add_middleware(ChatAuthMiddleware, container=container)
    return app


async def serve(settings: Settings) -> None:
    container = Container(settings)
    settings.require_server_tokens()
    set_container(container)

    await provision(container.database, container.qdrant, settings)
    await container.startup(run_relay=True)

    mcp_app = build_mcp_app(container)
    ops_app = build_ops_app(container)

    mcp_server = uvicorn.Server(
        uvicorn.Config(mcp_app, host=settings.mcp_host, port=settings.mcp_port, log_config=None)
    )
    ops_server = uvicorn.Server(
        uvicorn.Config(ops_app, host=settings.ops_host, port=settings.ops_port, log_config=None)
    )
    log.info(
        "server.serving",
        domain=settings.domain_id,
        mcp_port=settings.mcp_port,
        ops_port=settings.ops_port,
    )
    try:
        await asyncio.gather(mcp_server.serve(), ops_server.serve())
    finally:
        await container.shutdown()
