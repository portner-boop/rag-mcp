"""Enumerations for document/job state machines and pipeline stages (spec section 9)."""

from __future__ import annotations

from enum import Enum


class DocumentStatus(str, Enum):
    """Document lifecycle states.

    UPLOADING -> UPLOADED -> QUEUED -> PROCESSING -> READY
                                          -> FAILED
    UPLOADED|FAILED|READY -> REINDEXING -> READY
    UPLOADED|FAILED|READY -> DELETING -> DELETED | DELETE_FAILED
    """

    UPLOADING = "UPLOADING"
    UPLOADED = "UPLOADED"
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    READY = "READY"
    FAILED = "FAILED"
    REINDEXING = "REINDEXING"
    DELETING = "DELETING"
    DELETE_FAILED = "DELETE_FAILED"
    DELETED = "DELETED"


# Documents excluded from search (invariant 11).
SEARCH_EXCLUDED_STATUSES: frozenset[DocumentStatus] = frozenset(
    {DocumentStatus.DELETING, DocumentStatus.DELETE_FAILED, DocumentStatus.DELETED}
)


class JobStatus(str, Enum):
    """Common job envelope status (spec section 9)."""

    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    RETRY_WAIT = "RETRY_WAIT"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    DEAD_LETTER = "DEAD_LETTER"


class IngestionStage(str, Enum):
    """Ordered ingestion pipeline stages (spec section 9 and 11)."""

    DOWNLOAD = "DOWNLOAD"
    PARSING = "PARSING"
    MARKDOWN_UPLOAD = "MARKDOWN_UPLOAD"
    CHUNKING = "CHUNKING"
    DENSE_EMBEDDING = "DENSE_EMBEDDING"
    SPARSE_EMBEDDING = "SPARSE_EMBEDDING"
    QDRANT_UPSERT = "QDRANT_UPSERT"
    VERIFYING = "VERIFYING"
    FINALIZING = "FINALIZING"


# Progress percentage checkpoints per completed stage, used by get_ingestion_status.
STAGE_PROGRESS: dict[IngestionStage, int] = {
    IngestionStage.DOWNLOAD: 10,
    IngestionStage.PARSING: 25,
    IngestionStage.MARKDOWN_UPLOAD: 35,
    IngestionStage.CHUNKING: 45,
    IngestionStage.DENSE_EMBEDDING: 65,
    IngestionStage.SPARSE_EMBEDDING: 80,
    IngestionStage.QDRANT_UPSERT: 90,
    IngestionStage.VERIFYING: 96,
    IngestionStage.FINALIZING: 100,
}


class OutboxStatus(str, Enum):
    """Transactional outbox row status (spec section 8.2)."""

    PENDING = "PENDING"
    PUBLISHED = "PUBLISHED"
    FAILED = "FAILED"


class EventType(str, Enum):
    """Queue message / document event types (spec section 10)."""

    DOCUMENT_INGESTION_REQUESTED = "DocumentIngestionRequested"
    DOCUMENT_INGESTION_COMPLETED = "DocumentIngestionCompleted"
    DOCUMENT_INGESTION_FAILED = "DocumentIngestionFailed"
    DOCUMENT_DELETION_REQUESTED = "DocumentDeletionRequested"
    DOCUMENT_DELETED = "DocumentDeleted"
    DOCUMENT_DELETION_FAILED = "DocumentDeletionFailed"
    DOCUMENT_REINDEX_REQUESTED = "DocumentReindexRequested"
    DOCUMENT_REINDEX_COMPLETED = "DocumentReindexCompleted"
    DOCUMENT_REINDEX_FAILED = "DocumentReindexFailed"


class CapabilityProfile(str, Enum):
    """MCP capability profile (spec section 1)."""

    KNOWLEDGE = "KNOWLEDGE"
    HYBRID = "HYBRID"
