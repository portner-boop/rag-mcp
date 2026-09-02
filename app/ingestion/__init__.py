from app.ingestion.ports import (
    DocumentState,
    EmbeddingPort,
    IndexConfigState,
    IngestionStore,
    ObjectStorePort,
    PointData,
    VectorIndexPort,
)

__all__ = [
    "IngestionPipeline",
    "PipelineResult",
    "DocumentState",
    "IndexConfigState",
    "IngestionStore",
    "ObjectStorePort",
    "VectorIndexPort",
    "EmbeddingPort",
    "PointData",
]


def __getattr__(name: str):
    # Lazy re-export (PEP 562): importing an app.ingestion submodule (chunker,
    # table_linearize, ports) must not eagerly pull in `pipeline`, which imports
    # worker_support.chunk_embed and would close an import cycle back through this package.
    if name in ("IngestionPipeline", "PipelineResult"):
        from app.ingestion import pipeline

        return getattr(pipeline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
