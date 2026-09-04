from __future__ import annotations

from pydantic import Field, field_validator

from app.shared.contracts import StrictModel
from app.shared.enums import DocumentStatus


class SearchFilters(StrictModel):
    created_from: str | None = None
    created_to: str | None = None
    document_version: int | None = None


class SearchKnowledgeInput(StrictModel):
    query: str = Field(min_length=1, max_length=4000)
    limit: int = Field(default=10, ge=1, le=50)
    max_candidates: int = Field(default=50, ge=1, le=200)
    document_ids: list[str] | None = None
    filters: SearchFilters = Field(default_factory=SearchFilters)

    @field_validator("query")
    @classmethod
    def _non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value

    @field_validator("max_candidates")
    @classmethod
    def _candidates_ge_limit(cls, value: int, info) -> int:
        limit = info.data.get("limit")
        if limit is not None and value < limit:
            raise ValueError("max_candidates must be >= limit")
        return value


class SearchResult(StrictModel):
    source_id: str
    chunk_id: str
    document_id: str
    filename: str
    text: str
    page_from: int | None = None
    page_to: int | None = None
    section_path: list[str] = Field(default_factory=list)
    score: float
    index_version: int


class SearchMeta(StrictModel):
    dense_candidates: int
    sparse_candidates: int
    reranked: bool
    duration_ms: int
    expanded: bool = False
    top_dense_score: float | None = None
    top_rerank_score: float | None = None
    low_confidence: bool = False


class SearchKnowledgeOutput(StrictModel):
    query_id: str
    results: list[SearchResult]
    search_meta: SearchMeta


class FindDocumentsInput(StrictModel):
    query: str | None = None
    filename: str | None = None
    limit: int = Field(default=20, ge=1, le=100)
    status: DocumentStatus = DocumentStatus.READY


class DocumentRecord(StrictModel):
    document_id: str
    filename: str
    content_type: str
    size: int
    status: DocumentStatus
    document_version: int
    index_version: int | None = None
    chunk_count: int | None = None
    created_at: str
    indexed_at: str | None = None


class FindDocumentsOutput(StrictModel):
    documents: list[DocumentRecord]
    next_cursor: str | None = None


class GetDocumentMetadataInput(StrictModel):
    document_id: str


class DocumentMetadata(DocumentRecord):
    checksum: str | None = None
    parser_version: str | None = None
    chunker_version: str | None = None
    embedding_model: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    deleted_at: str | None = None
    updated_at: str | None = None


class PrepareUploadInput(StrictModel):
    filename: str = Field(min_length=1, max_length=512)
    content_type: str = Field(min_length=1, max_length=255)
    size: int = Field(gt=0)
    checksum: str | None = None
    created_by: str


class PrepareUploadOutput(StrictModel):
    document_id: str
    upload_url: str
    upload_headers: dict[str, str]
    expires_at: str


class StartIngestionInput(StrictModel):
    document_id: str
    checksum: str | None = None
    idempotency_key: str = Field(min_length=1)


class StartIngestionOutput(StrictModel):
    document_id: str
    job_id: str
    status: str


class GetIngestionStatusInput(StrictModel):
    job_id: str


class CancelJobInput(StrictModel):
    job_id: str


class CancelJobOutput(StrictModel):
    job_id: str
    cancel_requested: bool


class IngestionStatusOutput(StrictModel):
    job_id: str
    document_id: str
    status: str
    stage: str | None = None
    progress: int = 0
    attempt: int = 0
    error: dict | None = None


class DeleteDocumentInput(StrictModel):
    document_id: str
    requested_by: str
    idempotency_key: str = Field(min_length=1)


class DeleteDocumentOutput(StrictModel):
    document_id: str
    job_id: str
    status: str


class ReindexDocumentInput(StrictModel):
    document_id: str
    target_index_version: int = Field(ge=1)
    reason: str | None = None


class ReindexDocumentOutput(StrictModel):
    document_id: str
    job_id: str
    status: str
    target_index_version: int


class CreateDownloadUrlInput(StrictModel):
    document_id: str
    expires_in_seconds: int = Field(default=300, ge=1)


class CreateDownloadUrlOutput(StrictModel):
    document_id: str
    filename: str
    download_url: str
    expires_at: str
