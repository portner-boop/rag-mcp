from __future__ import annotations

from pydantic import ValidationError

from app.config import Settings

BASE = dict(
    domain_id="hr",
    postgres_url="postgresql+asyncpg://u:p@localhost/hr",
    qdrant_url="http://localhost:6333",
    qdrant_collection="hr-knowledge",
    s3_endpoint="http://localhost:9000",
    s3_bucket="hr-documents",
    s3_access_key="k",
    s3_secret_key="s",
    embedding_api_url="http://localhost:8000",
    rabbitmq_url="amqp://localhost",
    queue_namespace="hr",
    chat_service_token_hash="a" * 64,
    ops_service_token_hash="b" * 64,
)


def expect_fail(label: str, **overrides) -> None:
    cfg = {**BASE, **overrides}
    try:
        Settings(**cfg)  # type: ignore[arg-type]
    except (ValidationError, ValueError):
        print(f"  OK  startup rejected: {label}")
        return
    raise AssertionError(f"expected config failure for: {label}")


def main() -> None:
    print("config fail-fast smoke:")
    settings = Settings(**BASE)  # type: ignore[arg-type]
    assert settings.exchange_name == "domain.hr.commands"
    assert settings.queue_name("ingestion") == "domain.hr.ingestion"
    print("  OK  valid config accepted; topology derived")

    expect_fail("bucket not scoped to domain", s3_bucket="warehouse-documents")
    expect_fail("collection not scoped to domain", qdrant_collection="finance-knowledge")
    expect_fail("queue namespace mismatch", queue_namespace="legal")
    expect_fail("overlap >= chunk size", chunk_size_tokens=100, chunk_overlap_tokens=100)
    expect_fail("bad domain id", domain_id="HR!")

    settings.require_server_tokens()
    try:
        bad = Settings(**{**BASE, "ops_service_token_hash": "a" * 64})  # type: ignore[arg-type]
        bad.require_server_tokens()
    except ValueError:
        print("  OK  identical chat/ops token hashes rejected (invariant 2)")
    else:
        raise AssertionError("identical token hashes should be rejected")

    print("config smoke: PASS")


if __name__ == "__main__":
    main()
