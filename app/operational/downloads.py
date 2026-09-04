from __future__ import annotations

from datetime import timedelta

from app.config import Settings
from app.shared.contracts.mcp import CreateDownloadUrlInput, CreateDownloadUrlOutput
from app.shared.enums import SEARCH_EXCLUDED_STATUSES, DocumentStatus
from app.shared.errors import NotFoundError, ValidationError
from app.shared.time import to_rfc3339, utcnow
from app.storage.postgres.engine import Database
from app.storage.postgres.repositories import DocumentRepository
from app.storage.s3 import S3ObjectStore


class DownloadService:
    def __init__(
        self, *, database: Database, object_store: S3ObjectStore, settings: Settings
    ) -> None:
        self._db = database
        self._s3 = object_store
        self._settings = settings

    async def create_download_url(self, payload: CreateDownloadUrlInput) -> CreateDownloadUrlOutput:
        ttl = min(payload.expires_in_seconds, self._settings.presigned_download_max_ttl_seconds)
        async with self._db.session() as session:
            doc = await DocumentRepository(session).get_or_raise(payload.document_id)
            if DocumentStatus(doc.status) in SEARCH_EXCLUDED_STATUSES:
                raise ValidationError(
                    "Document is deleted or being deleted", details={"status": doc.status}
                )
            filename = doc.filename
            key = doc.original_object_key

        if not await self._s3.exists(key):
            raise NotFoundError("Original object is missing")

        url = await self._s3.presign_get(key, expires_in=ttl, download_filename=filename)
        return CreateDownloadUrlOutput(
            document_id=payload.document_id,
            filename=filename,
            download_url=url,
            expires_at=to_rfc3339(utcnow() + timedelta(seconds=ttl)),
        )
