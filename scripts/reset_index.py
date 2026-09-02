from __future__ import annotations

import argparse
import asyncio

from sqlalchemy import text

_TABLES = (
    "outbox_events",
    "inbox_events",
    "document_events",
    "ingestion_jobs",
    "deletion_jobs",
    "reindex_jobs",
    "document_versions",
    "documents",
    "index_configs",
)


async def reset() -> None:
    from app.container import build_container
    from app.provisioning import provision

    container = build_container()
    settings = container.settings

    try:
        await container.qdrant.drop_collection()
        print(f"dropped Qdrant collection {settings.qdrant_collection}")
    except Exception as exc:  # noqa: BLE001
        print(f"collection not dropped ({exc}); continuing")

    async with container.database.session() as session:
        await session.execute(text(f"TRUNCATE {', '.join(_TABLES)} RESTART IDENTITY CASCADE"))
        await session.commit()
    print(f"truncated {len(_TABLES)} tables in domain '{settings.domain_id}'")

    await provision(container.database, container.qdrant, settings)
    print(
        "provisioned index config v1: "
        f"{settings.embedding_dense_model} @ {settings.embedding_dense_dimension}, "
        f"sparse={settings.embedding_sparse_model}"
    )
    await container.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yes", action="store_true", help="required: this deletes the corpus")
    args = parser.parse_args()
    if not args.yes:
        parser.error("refusing to wipe the domain without --yes")
    asyncio.run(reset())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
