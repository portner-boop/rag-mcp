"""SQLAlchemy 2 ORM models for the domain PostgreSQL schema (spec section 8).

PostgreSQL stores metadata, object keys, jobs, events and the transactional
outbox/inbox only. Chunk text and vectors live exclusively in Qdrant (invariant 4).
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


def _uuid_col(primary_key: bool = False) -> Mapped[str]:
    return mapped_column(UUID(as_uuid=False), primary_key=primary_key)


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class Document(Base, TimestampMixin):
    __tablename__ = "documents"

    id: Mapped[str] = _uuid_col(primary_key=True)
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    content_type: Mapped[str] = mapped_column(String(255), nullable=False)
    size: Mapped[int] = mapped_column(BigInteger, nullable=False)
    checksum: Mapped[str | None] = mapped_column(String(128), nullable=True)
    original_object_key: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    markdown_object_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    document_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    index_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    chunk_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    parser_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    chunker_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_by: Mapped[str] = mapped_column(UUID(as_uuid=False), nullable=False)
    indexed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_code: Mapped[str | None] = mapped_column(String(128), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Optimistic-locking version counter for atomic state transitions (spec section 9).
    row_version: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    __table_args__ = (
        CheckConstraint("size > 0", name="ck_documents_size_positive"),
        Index("ix_documents_status_created", "status", "created_at"),
        Index("ix_documents_created_by_created", "created_by", "created_at"),
        Index("ix_documents_checksum_version", "checksum", "document_version"),
        Index("ix_documents_index_version", "index_version"),
        # The functional lower(filename) index is created in the Alembic migration.
    )


class DocumentVersion(Base):
    __tablename__ = "document_versions"

    id: Mapped[str] = _uuid_col(primary_key=True)
    document_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    checksum: Mapped[str | None] = mapped_column(String(128), nullable=True)
    original_object_key: Mapped[str] = mapped_column(Text, nullable=False)
    markdown_object_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    parser_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    chunker_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    source_size: Mapped[int] = mapped_column(BigInteger, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (UniqueConstraint("document_id", "version", name="uq_document_versions"),)


class JobMixin(TimestampMixin):
    """Common job envelope (spec section 8.2)."""

    id: Mapped[str] = _uuid_col(primary_key=True)
    document_id: Mapped[str] = mapped_column(UUID(as_uuid=False), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    stage: Mapped[str | None] = mapped_column(String(32), nullable=True)
    progress: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    attempt: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=5)
    idempotency_key: Mapped[str] = mapped_column(String(256), nullable=False)
    lease_owner: Mapped[str | None] = mapped_column(String(128), nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    cancel_requested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    available_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_code: Mapped[str | None] = mapped_column(String(128), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)


class IngestionJob(Base, JobMixin):
    __tablename__ = "ingestion_jobs"
    index_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    __table_args__ = (
        UniqueConstraint("idempotency_key", name="uq_ingestion_jobs_idem"),
        Index("ix_ingestion_jobs_status_created", "status", "created_at"),
        Index("ix_ingestion_jobs_document_created", "document_id", "created_at"),
        Index("ix_ingestion_jobs_lease", "lease_expires_at"),
    )


class DeletionJob(Base, JobMixin):
    __tablename__ = "deletion_jobs"
    requested_by: Mapped[str] = mapped_column(UUID(as_uuid=False), nullable=False)

    __table_args__ = (
        UniqueConstraint("idempotency_key", name="uq_deletion_jobs_idem"),
        Index("ix_deletion_jobs_status_created", "status", "created_at"),
        Index("ix_deletion_jobs_lease", "lease_expires_at"),
    )


class ReindexJob(Base, JobMixin):
    __tablename__ = "reindex_jobs"
    source_index_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    target_index_version: Mapped[int] = mapped_column(Integer, nullable=False)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        UniqueConstraint("idempotency_key", name="uq_reindex_jobs_idem"),
        UniqueConstraint("document_id", "target_index_version", name="uq_reindex_active_target"),
        Index("ix_reindex_jobs_status_created", "status", "created_at"),
        Index("ix_reindex_jobs_lease", "lease_expires_at"),
    )


class DocumentEvent(Base):
    __tablename__ = "document_events"

    id: Mapped[str] = _uuid_col(primary_key=True)
    document_id: Mapped[str] = mapped_column(UUID(as_uuid=False), nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    trace_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (Index("ix_document_events_doc_created", "document_id", "created_at"),)


class OutboxEvent(Base):
    __tablename__ = "outbox_events"

    event_id: Mapped[str] = _uuid_col(primary_key=True)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    aggregate_id: Mapped[str] = mapped_column(UUID(as_uuid=False), nullable=False)
    payload: Mapped[dict] = mapped_column(JSONB, nullable=False)
    routing_key: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="PENDING")
    attempt: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    available_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (Index("ix_outbox_status_available", "status", "available_at"),)


class InboxEvent(Base):
    __tablename__ = "inbox_events"

    consumer: Mapped[str] = mapped_column(String(128), primary_key=True)
    event_id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True)
    processed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    result_hash: Mapped[str | None] = mapped_column(String(128), nullable=True)


class IndexConfig(Base):
    __tablename__ = "index_configs"

    id: Mapped[str] = _uuid_col(primary_key=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False, unique=True)
    dense_model: Mapped[str] = mapped_column(String(255), nullable=False)
    dense_dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    sparse_model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    reranker_model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    chunk_size_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_overlap_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    qdrant_collection: Mapped[str] = mapped_column(String(255), nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        CheckConstraint(
            "chunk_overlap_tokens < chunk_size_tokens", name="ck_index_overlap_lt_size"
        ),
        CheckConstraint("dense_dimension > 0", name="ck_index_dense_dim_positive"),
        CheckConstraint("chunk_size_tokens > 0", name="ck_index_chunk_size_positive"),
        # "At most one active config" (spec 8.3) is enforced by a partial unique index
        # (WHERE active) created in the Alembic migration.
    )


__all__ = [
    "Base",
    "Document",
    "DocumentVersion",
    "IngestionJob",
    "DeletionJob",
    "ReindexJob",
    "DocumentEvent",
    "OutboxEvent",
    "InboxEvent",
    "IndexConfig",
]
