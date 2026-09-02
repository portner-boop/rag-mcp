from __future__ import annotations

from app.ingestion.ports import DocumentState
from app.storage.postgres.models import Document


def document_state_from(doc: Document) -> DocumentState:
    return DocumentState(
        document_id=doc.id,
        filename=doc.filename,
        content_type=doc.content_type,
        size=doc.size,
        status=doc.status,
        document_version=doc.document_version,
        checksum=doc.checksum,
        original_object_key=doc.original_object_key,
        markdown_object_key=doc.markdown_object_key,
        index_version=doc.index_version,
        chunk_count=doc.chunk_count,
    )
