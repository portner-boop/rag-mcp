"""Baseline domain schema: documents, versions, jobs, events, outbox/inbox, index configs.

Revision ID: 0001_baseline
Revises:
Create Date: 2026-08-28
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001_baseline"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

UUID = postgresql.UUID(as_uuid=False)
JSONB = postgresql.JSONB


def _job_columns() -> list[sa.Column]:
    return [
        sa.Column("id", UUID, primary_key=True),
        sa.Column("document_id", UUID, nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("stage", sa.String(32), nullable=True),
        sa.Column("progress", sa.Integer, nullable=False, server_default="0"),
        sa.Column("attempt", sa.Integer, nullable=False, server_default="0"),
        sa.Column("max_attempts", sa.Integer, nullable=False, server_default="5"),
        sa.Column("idempotency_key", sa.String(256), nullable=False),
        sa.Column("lease_owner", sa.String(128), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("cancel_requested_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("available_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_code", sa.String(128), nullable=True),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    ]


def upgrade() -> None:
    op.create_table(
        "documents",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("filename", sa.String(512), nullable=False),
        sa.Column("content_type", sa.String(255), nullable=False),
        sa.Column("size", sa.BigInteger, nullable=False),
        sa.Column("checksum", sa.String(128), nullable=True),
        sa.Column("original_object_key", sa.Text, nullable=False, unique=True),
        sa.Column("markdown_object_key", sa.Text, nullable=True),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("document_version", sa.Integer, nullable=False, server_default="1"),
        sa.Column("index_version", sa.Integer, nullable=True),
        sa.Column("chunk_count", sa.Integer, nullable=True),
        sa.Column("parser_version", sa.String(128), nullable=True),
        sa.Column("chunker_version", sa.String(128), nullable=True),
        sa.Column("embedding_model", sa.String(255), nullable=True),
        sa.Column("created_by", UUID, nullable=False),
        sa.Column("indexed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_code", sa.String(128), nullable=True),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("row_version", sa.Integer, nullable=False, server_default="0"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint("size > 0", name="ck_documents_size_positive"),
    )
    op.create_index("ix_documents_status_created", "documents", ["status", "created_at"])
    op.create_index("ix_documents_created_by_created", "documents", ["created_by", "created_at"])
    op.create_index("ix_documents_checksum_version", "documents", ["checksum", "document_version"])
    op.create_index("ix_documents_index_version", "documents", ["index_version"])
    # Functional lower(filename) index for case-insensitive filename search.
    op.execute("CREATE INDEX ix_documents_filename_lower ON documents (lower(filename))")

    op.create_table(
        "document_versions",
        sa.Column("id", UUID, primary_key=True),
        sa.Column(
            "document_id", UUID, sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("version", sa.Integer, nullable=False),
        sa.Column("checksum", sa.String(128), nullable=True),
        sa.Column("original_object_key", sa.Text, nullable=False),
        sa.Column("markdown_object_key", sa.Text, nullable=True),
        sa.Column("parser_version", sa.String(128), nullable=True),
        sa.Column("chunker_version", sa.String(128), nullable=True),
        sa.Column("source_size", sa.BigInteger, nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.UniqueConstraint("document_id", "version", name="uq_document_versions"),
    )

    op.create_table(
        "ingestion_jobs",
        *_job_columns(),
        sa.Column("index_version", sa.Integer, nullable=False, server_default="1"),
        sa.UniqueConstraint("idempotency_key", name="uq_ingestion_jobs_idem"),
    )
    op.create_index("ix_ingestion_jobs_status_created", "ingestion_jobs", ["status", "created_at"])
    op.create_index(
        "ix_ingestion_jobs_document_created", "ingestion_jobs", ["document_id", "created_at"]
    )
    op.create_index("ix_ingestion_jobs_lease", "ingestion_jobs", ["lease_expires_at"])

    op.create_table(
        "deletion_jobs",
        *_job_columns(),
        sa.Column("requested_by", UUID, nullable=False),
        sa.UniqueConstraint("idempotency_key", name="uq_deletion_jobs_idem"),
    )
    op.create_index("ix_deletion_jobs_status_created", "deletion_jobs", ["status", "created_at"])
    op.create_index("ix_deletion_jobs_lease", "deletion_jobs", ["lease_expires_at"])

    op.create_table(
        "reindex_jobs",
        *_job_columns(),
        sa.Column("source_index_version", sa.Integer, nullable=True),
        sa.Column("target_index_version", sa.Integer, nullable=False),
        sa.Column("reason", sa.Text, nullable=True),
        sa.UniqueConstraint("idempotency_key", name="uq_reindex_jobs_idem"),
        sa.UniqueConstraint("document_id", "target_index_version", name="uq_reindex_active_target"),
    )
    op.create_index("ix_reindex_jobs_status_created", "reindex_jobs", ["status", "created_at"])
    op.create_index("ix_reindex_jobs_lease", "reindex_jobs", ["lease_expires_at"])

    op.create_table(
        "document_events",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("document_id", UUID, nullable=False),
        sa.Column("event_type", sa.String(64), nullable=False),
        sa.Column("payload", JSONB, nullable=False),
        sa.Column("trace_id", sa.String(64), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )
    op.create_index(
        "ix_document_events_doc_created", "document_events", ["document_id", "created_at"]
    )

    op.create_table(
        "outbox_events",
        sa.Column("event_id", UUID, primary_key=True),
        sa.Column("event_type", sa.String(64), nullable=False),
        sa.Column("aggregate_id", UUID, nullable=False),
        sa.Column("payload", JSONB, nullable=False),
        sa.Column("routing_key", sa.String(255), nullable=False),
        sa.Column("status", sa.String(16), nullable=False, server_default="PENDING"),
        sa.Column("attempt", sa.Integer, nullable=False, server_default="0"),
        sa.Column(
            "available_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )
    op.create_index("ix_outbox_status_available", "outbox_events", ["status", "available_at"])

    op.create_table(
        "inbox_events",
        sa.Column("consumer", sa.String(128), primary_key=True),
        sa.Column("event_id", UUID, primary_key=True),
        sa.Column(
            "processed_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("result_hash", sa.String(128), nullable=True),
    )

    op.create_table(
        "index_configs",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("version", sa.Integer, nullable=False, unique=True),
        sa.Column("dense_model", sa.String(255), nullable=False),
        sa.Column("dense_dimension", sa.Integer, nullable=False),
        sa.Column("sparse_model", sa.String(255), nullable=True),
        sa.Column("reranker_model", sa.String(255), nullable=True),
        sa.Column("chunk_size_tokens", sa.Integer, nullable=False),
        sa.Column("chunk_overlap_tokens", sa.Integer, nullable=False),
        sa.Column("qdrant_collection", sa.String(255), nullable=False),
        sa.Column("active", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint(
            "chunk_overlap_tokens < chunk_size_tokens", name="ck_index_overlap_lt_size"
        ),
        sa.CheckConstraint("dense_dimension > 0", name="ck_index_dense_dim_positive"),
        sa.CheckConstraint("chunk_size_tokens > 0", name="ck_index_chunk_size_positive"),
    )
    # At most one active index config (spec section 8.3).
    op.execute(
        "CREATE UNIQUE INDEX uq_index_configs_single_active ON index_configs (active) WHERE active"
    )


def downgrade() -> None:
    op.drop_table("index_configs")
    op.drop_table("inbox_events")
    op.drop_table("outbox_events")
    op.drop_table("document_events")
    op.drop_table("reindex_jobs")
    op.drop_table("deletion_jobs")
    op.drop_table("ingestion_jobs")
    op.drop_table("document_versions")
    op.drop_index("ix_documents_filename_lower", table_name="documents")
    op.drop_table("documents")
