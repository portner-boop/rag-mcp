from __future__ import annotations

from typing import Any, Literal

from app.shared.contracts import SCHEMA_VERSION, StrictModel
from app.shared.enums import EventType


class EventEnvelope(StrictModel):
    schema_version: str = SCHEMA_VERSION
    event_id: str
    event_type: EventType
    occurred_at: str
    domain: str
    document_id: str
    job_id: str | None = None
    attempt: int = 0
    trace_id: str | None = None


class DocumentIngestionRequested(EventEnvelope):
    event_type: Literal[EventType.DOCUMENT_INGESTION_REQUESTED] = (
        EventType.DOCUMENT_INGESTION_REQUESTED
    )
    original_object_key: str
    index_version: int


class DocumentIngestionCompleted(EventEnvelope):
    event_type: Literal[EventType.DOCUMENT_INGESTION_COMPLETED] = (
        EventType.DOCUMENT_INGESTION_COMPLETED
    )
    chunk_count: int
    index_version: int
    duration_ms: int


class DocumentIngestionFailed(EventEnvelope):
    event_type: Literal[EventType.DOCUMENT_INGESTION_FAILED] = EventType.DOCUMENT_INGESTION_FAILED
    stage: str
    error_code: str
    retryable: bool


class DocumentDeletionRequested(EventEnvelope):
    event_type: Literal[EventType.DOCUMENT_DELETION_REQUESTED] = (
        EventType.DOCUMENT_DELETION_REQUESTED
    )
    requested_by: str


class DocumentDeleted(EventEnvelope):
    event_type: Literal[EventType.DOCUMENT_DELETED] = EventType.DOCUMENT_DELETED


class DocumentDeletionFailed(EventEnvelope):
    event_type: Literal[EventType.DOCUMENT_DELETION_FAILED] = EventType.DOCUMENT_DELETION_FAILED
    error_code: str
    retryable: bool


class DocumentReindexRequested(EventEnvelope):
    event_type: Literal[EventType.DOCUMENT_REINDEX_REQUESTED] = EventType.DOCUMENT_REINDEX_REQUESTED
    source_index_version: int | None = None
    target_index_version: int
    reason: str | None = None


class DocumentReindexCompleted(EventEnvelope):
    event_type: Literal[EventType.DOCUMENT_REINDEX_COMPLETED] = EventType.DOCUMENT_REINDEX_COMPLETED
    target_index_version: int
    chunk_count: int


class DocumentReindexFailed(EventEnvelope):
    event_type: Literal[EventType.DOCUMENT_REINDEX_FAILED] = EventType.DOCUMENT_REINDEX_FAILED
    target_index_version: int
    error_code: str
    retryable: bool


_EVENT_MODELS: dict[EventType, type[EventEnvelope]] = {
    EventType.DOCUMENT_INGESTION_REQUESTED: DocumentIngestionRequested,
    EventType.DOCUMENT_INGESTION_COMPLETED: DocumentIngestionCompleted,
    EventType.DOCUMENT_INGESTION_FAILED: DocumentIngestionFailed,
    EventType.DOCUMENT_DELETION_REQUESTED: DocumentDeletionRequested,
    EventType.DOCUMENT_DELETED: DocumentDeleted,
    EventType.DOCUMENT_DELETION_FAILED: DocumentDeletionFailed,
    EventType.DOCUMENT_REINDEX_REQUESTED: DocumentReindexRequested,
    EventType.DOCUMENT_REINDEX_COMPLETED: DocumentReindexCompleted,
    EventType.DOCUMENT_REINDEX_FAILED: DocumentReindexFailed,
}


def model_for(event_type: EventType) -> type[EventEnvelope]:
    return _EVENT_MODELS[event_type]


def parse_event(raw: dict[str, Any]) -> EventEnvelope:
    event_type = EventType(raw["event_type"])
    return model_for(event_type).model_validate(raw)
