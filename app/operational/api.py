"""Internal operational control-plane API (spec section 7.2-7.9).

Separate identity and network path from the chat MCP surface (invariant 2). Every route
requires the operational token and an explicit corpus scope. These operations are never
exposed to the chat orchestrator or the LLM. Delete/reindex land in the D04 slice.
"""

from __future__ import annotations

from fastapi import Depends, FastAPI, Header, Request
from fastapi.responses import JSONResponse, Response

from app.container import Container
from app.observability import metrics
from app.session.auth import Principal
from app.session.identity import Identity, Scope
from app.shared.contracts.mcp import (
    CancelJobInput,
    CancelJobOutput,
    CreateDownloadUrlInput,
    CreateDownloadUrlOutput,
    DeleteDocumentInput,
    DeleteDocumentOutput,
    DocumentMetadata,
    FindDocumentsInput,
    FindDocumentsOutput,
    GetDocumentMetadataInput,
    GetIngestionStatusInput,
    IngestionStatusOutput,
    PrepareUploadInput,
    PrepareUploadOutput,
    ReindexDocumentInput,
    ReindexDocumentOutput,
    StartIngestionInput,
    StartIngestionOutput,
)
from app.shared.errors import DomainError, InvalidStateError
from app.shared.trace import current_trace_id, parse_traceparent, set_current


def build_ops_app(container: Container) -> FastAPI:
    app = FastAPI(title=f"domain-mcp-ops-{container.settings.domain_id}", docs_url=None)

    def require_ops(scope: Scope):
        def _dep(
            authorization: str | None = Header(default=None),
            traceparent: str | None = Header(default=None),
            x_request_id: str | None = Header(default=None, alias="X-Request-ID"),
        ) -> Principal:
            set_current(parse_traceparent(traceparent, x_request_id))
            principal = container.authenticator.authenticate_identity(authorization, Identity.OPS)
            principal.require(scope)
            return principal

        return _dep

    @app.exception_handler(DomainError)
    async def _domain_error_handler(_request: Request, exc: DomainError) -> JSONResponse:
        return JSONResponse(exc.to_contract(current_trace_id()), status_code=exc.http_status)

    # --- 7.2 / 7.3 read ---------------------------------------------------------------
    @app.post("/internal/documents/find", response_model=FindDocumentsOutput)
    async def find_documents(
        payload: FindDocumentsInput, _: Principal = Depends(require_ops(Scope.CORPUS_READ))
    ) -> FindDocumentsOutput:
        return await container.document_service.find_documents(payload)

    @app.post("/internal/documents/get", response_model=DocumentMetadata)
    async def get_document_metadata(
        payload: GetDocumentMetadataInput,
        _: Principal = Depends(require_ops(Scope.CORPUS_READ)),
    ) -> DocumentMetadata:
        return await container.document_service.get_document_metadata(payload)

    # --- 7.4 / 7.5 / 7.6 ingestion ----------------------------------------------------
    @app.post("/internal/documents/prepare-upload", response_model=PrepareUploadOutput)
    async def prepare_upload(
        payload: PrepareUploadInput, _: Principal = Depends(require_ops(Scope.CORPUS_WRITE))
    ) -> PrepareUploadOutput:
        return await container.ingestion_service.prepare_document_upload(payload)

    @app.post("/internal/ingestion/start", response_model=StartIngestionOutput)
    async def start_ingestion(
        payload: StartIngestionInput, _: Principal = Depends(require_ops(Scope.CORPUS_WRITE))
    ) -> StartIngestionOutput:
        return await container.ingestion_service.start_document_ingestion(payload)

    @app.post("/internal/ingestion/status", response_model=IngestionStatusOutput)
    async def ingestion_status(
        payload: GetIngestionStatusInput,
        _: Principal = Depends(require_ops(Scope.CORPUS_READ)),
    ) -> IngestionStatusOutput:
        return await container.ingestion_service.get_ingestion_status(payload)

    # --- 7.7 delete / 7.8 reindex (D04) -----------------------------------------------
    @app.post("/internal/documents/delete", response_model=DeleteDocumentOutput)
    async def delete_document(
        payload: DeleteDocumentInput, _: Principal = Depends(require_ops(Scope.CORPUS_DELETE))
    ) -> DeleteDocumentOutput:
        return await container.lifecycle_service.delete_document(payload)

    @app.post("/internal/documents/reindex", response_model=ReindexDocumentOutput)
    async def reindex_document(
        payload: ReindexDocumentInput, _: Principal = Depends(require_ops(Scope.CORPUS_REINDEX))
    ) -> ReindexDocumentOutput:
        return await container.lifecycle_service.reindex_document(payload)

    @app.post("/internal/jobs/cancel", response_model=CancelJobOutput)
    async def cancel_job(
        payload: CancelJobInput, _: Principal = Depends(require_ops(Scope.CORPUS_WRITE))
    ) -> CancelJobOutput:
        return await container.lifecycle_service.cancel_job(payload)

    # --- DLQ inspect / redrive (D04) --------------------------------------------------
    @app.post("/internal/dlq/inspect")
    async def dlq_inspect(
        limit: int = 20, _: Principal = Depends(require_ops(Scope.CORPUS_READ))
    ) -> JSONResponse:
        if container.dlq_service is None:
            raise InvalidStateError("Queue is not connected")
        return JSONResponse({"messages": await container.dlq_service.inspect(limit)})

    @app.post("/internal/dlq/redrive")
    async def dlq_redrive(
        limit: int = 20, _: Principal = Depends(require_ops(Scope.CORPUS_WRITE))
    ) -> JSONResponse:
        if container.dlq_service is None:
            raise InvalidStateError("Queue is not connected")
        return JSONResponse({"redriven": await container.dlq_service.redrive(limit)})

    # --- 7.9 download -----------------------------------------------------------------
    @app.post("/internal/downloads/create", response_model=CreateDownloadUrlOutput)
    async def create_download_url(
        payload: CreateDownloadUrlInput,
        _: Principal = Depends(require_ops(Scope.CORPUS_READ)),
    ) -> CreateDownloadUrlOutput:
        return await container.download_service.create_download_url(payload)

    # --- health / metrics -------------------------------------------------------------
    @app.get("/health/operational")
    async def operational_health() -> JSONResponse:
        report = await container.operational_health()
        return JSONResponse(
            {"status": "ok" if report.ok else "degraded", "checks": report.checks},
            status_code=200 if report.ok else 503,
        )

    @app.get("/metrics")
    async def prometheus_metrics() -> Response:
        body, content_type = metrics.render_latest()
        return Response(content=body, media_type=content_type)

    return app
