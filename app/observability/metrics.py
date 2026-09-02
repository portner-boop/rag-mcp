"""Prometheus metrics (spec section 17).

A single default registry is shared across the process. The worker reuses the same
module so worker + server metric names stay consistent.
"""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

# --- MCP / search metrics ---
mcp_tool_calls_total = Counter("mcp_tool_calls_total", "MCP tool calls", ["tool", "status"])
mcp_tool_duration_seconds = Histogram("mcp_tool_duration_seconds", "MCP tool duration", ["tool"])
search_requests_total = Counter("search_requests_total", "Search requests", ["status"])
search_duration_seconds = Histogram("search_duration_seconds", "Search duration")
search_results_count = Histogram("search_results_count", "Results per search")
search_empty_results_total = Counter("search_empty_results_total", "Empty searches")
dense_search_duration_seconds = Histogram("dense_search_duration_seconds", "Dense search")
sparse_search_duration_seconds = Histogram("sparse_search_duration_seconds", "Sparse search")
rerank_duration_seconds = Histogram("rerank_duration_seconds", "Rerank duration")
rerank_fallback_total = Counter("rerank_fallback_total", "Rerank fallbacks")
documents_by_status = Gauge("documents_by_status", "Documents by status", ["status"])

# --- worker / ingestion metrics ---
ingestion_jobs_total = Counter("ingestion_jobs_total", "Ingestion jobs", ["result"])
ingestion_jobs_failed_total = Counter("ingestion_jobs_failed_total", "Failed ingestion jobs")
ingestion_job_duration_seconds = Histogram(
    "ingestion_job_duration_seconds", "Ingestion job duration"
)
ingestion_stage_duration_seconds = Histogram(
    "ingestion_stage_duration_seconds", "Ingestion stage duration", ["stage"]
)
ingestion_retries_total = Counter("ingestion_retries_total", "Ingestion retries")
ingestion_dlq_total = Counter("ingestion_dlq_total", "Ingestion DLQ")
chunks_created_total = Counter("chunks_created_total", "Chunks created")
qdrant_points_upserted_total = Counter("qdrant_points_upserted_total", "Qdrant points upserted")

# --- generic worker metrics (deletion / reindex / recovery) ---
worker_jobs_total = Counter("worker_jobs_total", "Worker jobs", ["consumer", "result"])
worker_retries_total = Counter("worker_retries_total", "Worker retries", ["consumer"])
worker_dlq_total = Counter("worker_dlq_total", "Worker DLQ", ["consumer"])
recovery_actions_total = Counter("recovery_actions_total", "Recovery actions", ["kind"])


def render_latest() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
