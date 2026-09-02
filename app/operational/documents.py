"""Internal document read use cases: find_documents, get_document_metadata (spec 7.2/7.3)."""

from __future__ import annotations

from app.operational.mappers import to_metadata, to_record
from app.shared.contracts.mcp import (
    DocumentMetadata,
    FindDocumentsInput,
    FindDocumentsOutput,
    GetDocumentMetadataInput,
)
from app.storage.postgres.engine import Database
from app.storage.postgres.repositories import DocumentRepository


class DocumentReadService:
    def __init__(self, database: Database) -> None:
        self._db = database

    async def find_documents(self, payload: FindDocumentsInput) -> FindDocumentsOutput:
        async with self._db.session() as session:
            repo = DocumentRepository(session)
            docs = await repo.find(
                status=payload.status,
                query=payload.query,
                filename=payload.filename,
                limit=payload.limit,
            )
            return FindDocumentsOutput(documents=[to_record(d) for d in docs], next_cursor=None)

    async def get_document_metadata(self, payload: GetDocumentMetadataInput) -> DocumentMetadata:
        async with self._db.session() as session:
            repo = DocumentRepository(session)
            doc = await repo.get_or_raise(payload.document_id)
            return to_metadata(doc)
