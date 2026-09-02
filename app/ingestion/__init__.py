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
    if name in ("IngestionPipeline", "PipelineResult"):
        from app.ingestion import pipeline

        return getattr(pipeline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
