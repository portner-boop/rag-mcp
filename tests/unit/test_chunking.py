"""Deterministic chunker: order, overlap, section paths, pages (spec sections 9, 11)."""

from __future__ import annotations

import pytest

from app.ingestion.chunker import DeterministicChunker


def test_overlap_shares_words_between_consecutive_chunks() -> None:
    words = " ".join(f"w{i}" for i in range(200))
    chunker = DeterministicChunker(chunk_size_tokens=40, chunk_overlap_tokens=10)
    chunks = chunker.chunk(words)
    assert len(chunks) > 1
    for a, b in zip(chunks, chunks[1:], strict=False):
        tail = a.text.split()[-10:]
        head = b.text.split()[:10]
        assert tail == head  # the overlap window is shared verbatim


def test_section_path_tracks_markdown_headings() -> None:
    md = "# Top\n\nintro words here\n## Sub\n\n" + ("body " * 50)
    chunks = DeterministicChunker(chunk_size_tokens=20, chunk_overlap_tokens=5).chunk(md)
    assert chunks[0].section_path == ["Top"]
    assert any(c.section_path == ["Top", "Sub"] for c in chunks)


def test_pages_mapped_from_offsets() -> None:
    md = ("alpha " * 30) + ("beta " * 30)
    page_offsets = [len("alpha " * 30), len(md)]  # page 1 ends mid-text, page 2 to end
    chunks = DeterministicChunker(chunk_size_tokens=20, chunk_overlap_tokens=0).chunk(
        md, page_offsets=page_offsets
    )
    assert chunks[0].page_from == 1
    assert chunks[-1].page_to == 2


def test_empty_markdown_yields_no_chunks() -> None:
    assert DeterministicChunker(chunk_size_tokens=40, chunk_overlap_tokens=8).chunk("   \n\n") == []


def test_overlap_must_be_less_than_size() -> None:
    with pytest.raises(ValueError):
        DeterministicChunker(chunk_size_tokens=10, chunk_overlap_tokens=10)


# --- table awareness (search-improvement S1, Tier 1.2 / 1.4) --------------------------


def test_table_is_an_atomic_chunk_not_merged_with_prose() -> None:
    md = (
        "## Costs\n\n"
        "some prose before the table here\n\n"
        "| Item | Price |\n|---|---|\n| A | 10 |\n| B | 20 |\n\n"
        "some prose after the table here\n"
    )
    chunks = DeterministicChunker(chunk_size_tokens=40, chunk_overlap_tokens=8).chunk(md)
    table_chunks = [c for c in chunks if c.text.lstrip().startswith("|")]
    assert len(table_chunks) == 1
    table = table_chunks[0]
    # The whole table stays together with its header, and no prose bleeds in.
    assert "| Item | Price |" in table.text and "| B | 20 |" in table.text
    assert "prose" not in table.text
    # Section path reflects where the table actually sits (Tier 1.4).
    assert table.section_path == ["Costs"]


def test_large_table_splits_by_rows_repeating_the_header() -> None:
    rows = "\n".join(f"| r{i} | {i} |" for i in range(40))
    md = "| Name | Value |\n|---|---|\n" + rows + "\n"
    chunks = DeterministicChunker(chunk_size_tokens=30, chunk_overlap_tokens=5).chunk(md)
    table_chunks = [c for c in chunks if c.text.lstrip().startswith("|")]
    assert len(table_chunks) > 1  # too big for one window
    for c in table_chunks:
        assert c.text.startswith("| Name | Value |")  # header repeated in every part


def test_table_free_markdown_is_unchanged_single_run() -> None:
    md = "# Doc\n\n" + " ".join(f"w{i}" for i in range(120))
    chunks = DeterministicChunker(chunk_size_tokens=40, chunk_overlap_tokens=10).chunk(md)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert all(c.section_path == ["Doc"] for c in chunks)
