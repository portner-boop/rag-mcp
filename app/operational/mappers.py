"""Map ORM rows to boundary contracts (spec section 7).

Object keys are internal and never returned to frontend-facing clients (spec 7.3).
"""

from __future__ import annotations

from app.shared.contracts.mcp import DocumentMetadata, DocumentRecord
from app.shared.enums import DocumentStatus
from app.shared.time import to_rfc3339
from app.storage.postgres.models import Document


def to_record(doc: Document) -> DocumentRecord:
    return DocumentRecord(
        document_id=doc.id,
        filename=doc.filename,
        content_type=doc.content_type,
        size=doc.size,
        status=DocumentStatus(doc.status),
        document_version=doc.document_version,
        index_version=doc.index_version,
        chunk_count=doc.chunk_count,
        created_at=to_rfc3339(doc.created_at),
        indexed_at=to_rfc3339(doc.indexed_at) if doc.indexed_at else None,
    )


def to_metadata(doc: Document) -> DocumentMetadata:
    return DocumentMetadata(
        document_id=doc.id,
        filename=doc.filename,
        content_type=doc.content_type,
        size=doc.size,
        status=DocumentStatus(doc.status),
        document_version=doc.document_version,
        index_version=doc.index_version,
        chunk_count=doc.chunk_count,
        created_at=to_rfc3339(doc.created_at),
        indexed_at=to_rfc3339(doc.indexed_at) if doc.indexed_at else None,
        checksum=doc.checksum,
        parser_version=doc.parser_version,
        chunker_version=doc.chunker_version,
        embedding_model=doc.embedding_model,
        error_code=doc.error_code,
        error_message=doc.error_message,
        deleted_at=to_rfc3339(doc.deleted_at) if doc.deleted_at else None,
        updated_at=to_rfc3339(doc.updated_at) if doc.updated_at else None,
    )
