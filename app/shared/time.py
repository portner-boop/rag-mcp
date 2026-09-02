"""UTC time helpers. Timestamps are always UTC RFC 3339 (spec section 7)."""

from __future__ import annotations

from datetime import UTC, datetime


def utcnow() -> datetime:
    """Timezone-aware current UTC time."""
    return datetime.now(UTC)


def to_rfc3339(value: datetime) -> str:
    """Serialize a datetime to RFC 3339 with a trailing Z for UTC."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    value = value.astimezone(UTC)
    return value.isoformat().replace("+00:00", "Z")
