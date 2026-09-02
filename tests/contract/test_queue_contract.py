"""Queue envelope/event contract and round-trip (spec section 10)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.shared.contracts.queue import (
    DocumentDeleted,
    DocumentDeletionRequested,
    DocumentIngestionCompleted,
    DocumentIngestionFailed,
    DocumentIngestionRequested,
    DocumentReindexCompleted,
    DocumentReindexRequested,
    parse_event,
)
from app.shared.enums import EventType

_COMMON = dict(event_id="e", occurred_at="2026-01-01T00:00:00Z", domain="hr", document_id="d")


def _events():
    return [
        DocumentIngestionRequested(**_COMMON, job_id="j", original_object_key="k", index_version=1),
        DocumentIngestionCompleted(
            **_COMMON, job_id="j", chunk_count=3, index_version=1, duration_ms=9
        ),
        DocumentIngestionFailed(
            **_COMMON, job_id="j", stage="PARSING", error_code="X", retryable=True
        ),
        DocumentDeletionRequested(**_COMMON, job_id="j", requested_by="admin"),
        DocumentDeleted(**_COMMON, job_id="j"),
        DocumentReindexRequested(
            **_COMMON, job_id="j", source_index_version=1, target_index_version=2
        ),
        DocumentReindexCompleted(**_COMMON, job_id="j", target_index_version=2, chunk_count=3),
    ]


@pytest.mark.parametrize("event", _events())
def test_event_roundtrips_through_parse(event) -> None:
    raw = event.model_dump(mode="json")
    parsed = parse_event(raw)
    assert type(parsed) is type(event)
    assert parsed.event_id == event.event_id
    assert parsed.domain == event.domain


def test_envelope_has_schema_version_default() -> None:
    e = DocumentDeleted(**_COMMON, job_id="j")
    assert e.schema_version == "1.3.0"
    assert e.event_type == EventType.DOCUMENT_DELETED


def test_parse_rejects_unknown_event_type() -> None:
    with pytest.raises((ValueError, KeyError)):
        parse_event({**_COMMON, "event_type": "NotARealEvent"})


def test_parse_rejects_missing_required_field() -> None:
    # DocumentIngestionRequested requires original_object_key + index_version.
    with pytest.raises(ValidationError):
        parse_event({**_COMMON, "event_type": "DocumentIngestionRequested", "job_id": "j"})
