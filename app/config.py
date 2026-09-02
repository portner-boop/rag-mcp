"""Configuration: Dynaconf loads `settings.toml` + `RAG_MCP_` env, validated by a
Pydantic model that enforces fail-fast startup and the domain isolation guard.

Architecture mirrors bpmn-mcp (`Dynaconf(envvar_prefix=..., settings_files=[...])`),
but the loaded values are validated through `Settings` so a bucket/queue/collection not
scoped to `DOMAIN_ID`, or a missing required value, still stops startup (spec section 5).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from dynaconf import Dynaconf
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

from app.shared.enums import CapabilityProfile

_ROOT = Path(__file__).resolve().parent.parent

settings = Dynaconf(
    envvar_prefix="RAG_MCP",
    root_path=str(_ROOT),
    settings_files=["settings.toml"],
    environments=False,
    load_dotenv=True,
)


def _scoped_to_domain(value: str, domain_id: str) -> bool:
    """A store/queue name is isolated only if the domain id appears as a bounded token."""
    seps = "-_.:/ "
    padded = f"{seps[0]}{value}{seps[0]}"
    for sep_left in seps:
        for sep_right in seps:
            if f"{sep_left}{domain_id}{sep_right}" in padded:
                return True
    return value == domain_id


class Settings(BaseModel):
    """Validated runtime configuration for both server and worker modes."""

    model_config = ConfigDict(extra="ignore")

    # --- identity ---
    domain_id: str = Field(min_length=1)
    capability_profile: CapabilityProfile = CapabilityProfile.KNOWLEDGE
    contract_version: str = "1.3.0"
    environment: str = "local"

    # --- postgres ---
    postgres_url: str = Field(min_length=1)

    # --- qdrant ---
    qdrant_url: str = Field(min_length=1)
    qdrant_api_key: str = ""
    qdrant_collection: str = Field(min_length=1)

    # --- s3 ---
    s3_endpoint: str = Field(min_length=1)
    s3_region: str = "us-east-1"
    s3_bucket: str = Field(min_length=1)
    s3_access_key: str = Field(min_length=1)
    s3_secret_key: str = Field(min_length=1)
    s3_use_path_style: bool = True

    # --- auth (server only) ---
    chat_service_token_hash: str = ""
    ops_service_token_hash: str = ""

    # --- embedding api ---
    embedding_api_url: str = Field(min_length=1)
    embedding_api_token: str = "local-placeholder"
    embedding_dense_model: str = "qwen/qwen3-embedding-8b"
    embedding_sparse_model: str = "qdrant/bm25"
    embedding_reranker_model: str = "qwen/qwen3-reranker-4b"
    embedding_dense_dimension: int = 4096

    # --- retrieval backends ---
    # Dense + rerank transport:
    #   "openai" — OpenAI-shaped POST /embeddings and Cohere-shaped POST /rerank, i.e.
    #              OpenRouter (embedding_api_url = https://openrouter.ai/api/v1) or any
    #              compatible gateway;
    #   "custom" — this stack's own Embedding API (/v1/embeddings/dense, /v1/rerank).
    embedding_provider: str = "openai"
    # Lexical branch:
    #   "bm25" — BM25 term weights computed in-process; Qdrant supplies IDF through the
    #            sparse index `modifier: idf`, which is its built-in BM25 scoring;
    #   "api"  — a sparse embedding model served by the custom Embedding API.
    sparse_provider: str = "bm25"
    # Qwen3 embedding models are instruction-tuned on the query side only; documents are
    # embedded verbatim. Empty string disables the prefix.
    embedding_query_instruction: str = (
        "Instruct: Given a search query, retrieve relevant passages that answer it\nQuery: "
    )

    # --- bm25 (sparse lexical branch) ---
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    bm25_avg_doc_len_tokens: int = 400
    bm25_vocab_size: int = 1 << 20

    # --- queue ---
    rabbitmq_url: str = Field(min_length=1)
    queue_namespace: str = Field(min_length=1)

    # --- upload / presign limits ---
    upload_max_bytes: int = 50 * 1024 * 1024
    allowed_content_types: tuple[str, ...] = (
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/plain",
        "text/markdown",
    )
    presigned_upload_ttl_seconds: int = 900
    presigned_download_ttl_seconds: int = 300
    presigned_download_max_ttl_seconds: int = 3600

    # --- mcp / transport ---
    mcp_host: str = "127.0.0.1"
    mcp_port: int = 8080
    mcp_path: str = "/mcp"
    ops_host: str = "127.0.0.1"
    ops_port: int = 8090
    mcp_tool_timeout_seconds: float = 30.0
    mcp_max_response_bytes: int = 1_000_000

    # --- parser / chunker ---
    parser_timeout_seconds: float = 60.0
    parser_max_pages: int = 2000
    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 64

    # --- embedding batching ---
    embedding_batch_size: int = 32
    embedding_concurrency: int = 4
    embedding_timeout_seconds: float = 60.0
    rerank_timeout_seconds: float = 30.0

    # --- search ---
    search_dense_candidates: int = 50
    search_sparse_candidates: int = 50
    search_default_limit: int = 10
    search_max_limit: int = 50
    search_max_candidates: int = 200
    enable_reranker: bool = True
    allow_dense_only_fallback: bool = True
    allow_sparse_only_fallback: bool = False
    search_rrf_k: int = 60
    search_max_chunk_chars: int = 4000
    # When a hit is a table row/fact, return the whole table (header + rows) so the LLM can
    # read the answer cell in context (search-improvement S2, Tier 4.1). Bounded by the char
    # limit above.
    search_return_full_table: bool = True

    # --- query expansion (search-improvement S2, Tier 3.1) ---
    # Off by default: expansion is a *selective fallback* run only when the first-pass rerank
    # is low-confidence, because HyDE costs an extra LLM call and each variant costs an extra
    # query embedding + retrieval. It never fires on the confident common case.
    enable_query_expansion: bool = False
    # Trigger when the top rerank score is below this (rerank-score scale, typically 0..1).
    # Requires the reranker to run — it is the confidence signal.
    query_expansion_min_rerank_score: float = 0.30
    # How many paraphrase variants to request (a HyDE passage is added on top when enabled).
    query_expansion_max_variants: int = 3
    enable_hyde: bool = True
    # Generation model + timeout for the expander (same OpenAI-shaped gateway as embeddings).
    expansion_model: str = "qwen/qwen3-8b"
    expansion_timeout_seconds: float = 15.0

    # --- worker ---
    worker_concurrency: int = 4
    worker_prefetch: int = 8
    lease_ttl_seconds: int = 120
    heartbeat_interval_seconds: int = 30
    max_attempts: int = 5
    retry_schedule_seconds: tuple[int, ...] = (0, 1, 3, 10, 30)
    retry_jitter_seconds: float = 1.0
    stale_upload_ttl_seconds: int = 3600
    recovery_interval_seconds: int = 60

    # --- queue timeouts ---
    qdrant_timeout_seconds: float = 10.0
    s3_metadata_timeout_seconds: float = 5.0

    # --- logging ---
    log_level: str = "INFO"

    @field_validator("domain_id")
    @classmethod
    def _validate_domain(cls, value: str) -> str:
        import re

        if not re.fullmatch(r"[a-z][a-z0-9_-]{1,62}", value):
            raise ValueError("DOMAIN_ID must be lowercase alphanumeric with '-'/'_' (2-63 chars)")
        return value

    @field_validator("s3_bucket", "qdrant_collection", "queue_namespace")
    @classmethod
    def _validate_isolation(cls, value: str, info: ValidationInfo) -> str:
        domain_id = info.data.get("domain_id")
        if domain_id and not _scoped_to_domain(value, domain_id):
            raise ValueError(
                f"{info.field_name}='{value}' is not scoped to DOMAIN_ID='{domain_id}'; "
                "bucket/collection/queue names must contain the domain id (invariant 1/3)"
            )
        return value

    @field_validator("embedding_provider")
    @classmethod
    def _validate_embedding_provider(cls, value: str) -> str:
        if value not in ("openai", "custom"):
            raise ValueError("embedding_provider must be 'openai' or 'custom'")
        return value

    @field_validator("sparse_provider")
    @classmethod
    def _validate_sparse_provider(cls, value: str) -> str:
        if value not in ("bm25", "api"):
            raise ValueError("sparse_provider must be 'bm25' or 'api'")
        return value

    @model_validator(mode="after")
    def _validate_retrieval_backends(self) -> Settings:
        if self.embedding_provider == "openai" and self.sparse_provider == "api":
            raise ValueError(
                "sparse_provider='api' needs embedding_provider='custom': the OpenAI-shaped "
                "API has no sparse embedding endpoint (use sparse_provider='bm25')"
            )
        if self.bm25_avg_doc_len_tokens <= 0:
            raise ValueError("bm25_avg_doc_len_tokens must be positive")
        if self.bm25_vocab_size < 1024:
            raise ValueError("bm25_vocab_size must be at least 1024")
        return self

    @model_validator(mode="after")
    def _validate_chunking(self) -> Settings:
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ValueError("chunk_overlap_tokens must be < chunk_size_tokens")
        if self.embedding_dense_dimension <= 0:
            raise ValueError("embedding_dense_dimension must be positive")
        return self

    # --- derived queue topology (spec section 10) ---
    @property
    def exchange_name(self) -> str:
        return f"domain.{self.queue_namespace}.commands"

    @property
    def dlx_name(self) -> str:
        return f"domain.{self.queue_namespace}.dlx"

    @property
    def dlq_name(self) -> str:
        return f"domain.{self.queue_namespace}.dlq"

    def queue_name(self, kind: str) -> str:
        return f"domain.{self.queue_namespace}.{kind}"

    def routing_key(self, kind: str) -> str:
        return f"{self.queue_namespace}.{kind}"

    def require_server_tokens(self) -> None:
        """Server-only fail-fast: both token hashes must be present and distinct."""
        if not self.chat_service_token_hash or not self.ops_service_token_hash:
            raise ValueError(
                "CHAT_SERVICE_TOKEN_HASH and OPS_SERVICE_TOKEN_HASH are required for the server"
            )
        if self.chat_service_token_hash == self.ops_service_token_hash:
            raise ValueError("Chat and ops service token hashes must be distinct (invariant 2)")


@lru_cache
def get_settings() -> Settings:
    """Build the validated Settings from Dynaconf (settings.toml + RAG_MCP_ env + .env)."""
    fields = set(Settings.model_fields)
    data = {
        key.lower(): value for key, value in settings.as_dict().items() if key.lower() in fields
    }
    return Settings(**data)
