from __future__ import annotations

from collections.abc import Callable

from app.ingestion.chunker import Chunk, DeterministicChunker
from app.ingestion.ports import PointData
from app.ingestion.table_linearize import linearize_tables
from app.shared.time import to_rfc3339, utcnow

TABLE_FACT_INDEX_BASE = 1_000_000
_TABLE_FACTS_VERSION = "tablefacts-1"


def chunk_markdown(
    markdown: str,
    *,
    page_offsets: list[int] | None,
    chunk_size_tokens: int,
    chunk_overlap_tokens: int,
) -> tuple[list[Chunk], str]:
    chunker = DeterministicChunker(
        chunk_size_tokens=chunk_size_tokens, chunk_overlap_tokens=chunk_overlap_tokens
    )
    chunks = chunker.chunk(markdown, page_offsets=page_offsets)
    facts = linearize_tables(markdown)

    table_markdown_by_index = {fact.table_index: fact.table_markdown for fact in facts}
    for chunk in chunks:
        if chunk.table_index is not None and chunk.table_markdown is None:
            chunk.table_markdown = table_markdown_by_index.get(chunk.table_index)

    for ordinal, fact in enumerate(facts):
        chunks.append(
            Chunk(
                chunk_index=TABLE_FACT_INDEX_BASE + ordinal,
                text=fact.text,
                section_path=list(fact.section_path),
                page_from=None,
                page_to=None,
                token_count=len(fact.text.split()),
                table_index=fact.table_index,
                table_markdown=fact.table_markdown,
            )
        )
    version = f"{chunker.version}+{_TABLE_FACTS_VERSION}"
    return chunks, version


def embed_texts(chunks: list[Chunk], *, filename: str) -> list[str]:
    return [_embed_text(chunk, filename) for chunk in chunks]


def _embed_text(chunk: Chunk, filename: str) -> str:
    parts = [filename] if filename else []
    parts.extend(part for part in chunk.section_path if part)
    prefix = " › ".join(parts)
    return f"{prefix}\n\n{chunk.text}" if prefix else chunk.text


def build_points(
    *,
    document_id: str,
    filename: str,
    content_type: str,
    document_version: int,
    index_version: int,
    chunks: list[Chunk],
    dense: list[list[float]],
    sparse,
    point_id: Callable[[int], str],
) -> list[PointData]:
    created_at = to_rfc3339(utcnow())
    points: list[PointData] = []
    for chunk, dvec, svec in zip(chunks, dense, sparse, strict=True):
        payload = {
            "document_id": document_id,
            "filename": filename,
            "text": chunk.text,
            "chunk_index": chunk.chunk_index,
            "page_from": chunk.page_from,
            "page_to": chunk.page_to,
            "section_path": chunk.section_path,
            "document_version": document_version,
            "index_version": index_version,
            "content_type": content_type,
            "created_at": created_at,
        }
        if chunk.table_index is not None:
            payload["table_id"] = (
                f"{document_id}:{document_version}:{index_version}:t{chunk.table_index}"
            )
            payload["table_markdown"] = chunk.table_markdown
        points.append(
            PointData(
                id=point_id(chunk.chunk_index),
                dense=dvec,
                sparse_indices=svec.indices,
                sparse_values=svec.values,
                payload=payload,
            )
        )
    return points
