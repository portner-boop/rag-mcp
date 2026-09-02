"""Deterministic Markdown chunker (spec sections 9, 11, invariant 9).

Given the same Markdown and config it always produces the same ordered chunks with the
same section paths and page ranges, so stable point IDs derived from
(document, version, index, chunk_index) are reproducible on re-ingestion.

Chunking is **block-aware** (search-improvement S1, Tier 1.2): prose is split into
whitespace-word windows with overlap, but a Markdown table is an atomic block — it is never
merged with surrounding prose and never split across a window boundary, so a value keeps its
column header and a chunk's ``section_path`` reflects where its content actually sits (Tier
1.4). A table larger than one window is split by data rows with the header row repeated in
every part. Markdown with no tables chunks exactly as a single prose run.

Tokens are approximated by whitespace words to stay dependency-free and fully
deterministic; a production tokenizer can replace the word split without changing IDs or
ordering. Section path comes from the Markdown ATX headings in scope at a chunk's start.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.ingestion.table_linearize import table_line_ranges

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_WORD_RE = re.compile(r"\S+")


@dataclass
class Chunk:
    chunk_index: int
    text: str
    section_path: list[str] = field(default_factory=list)
    page_from: int | None = None
    page_to: int | None = None
    token_count: int = 0
    # Set on table-derived chunks (search-improvement S2, Tier 4.1): the 0-based table index
    # within the document and the whole-table Markdown, so search can return the full table.
    table_index: int | None = None
    table_markdown: str | None = None


@dataclass
class _Word:
    text: str
    start: int
    end: int
    section_path: tuple[str, ...]


class DeterministicChunker:
    version = "chunker-2"

    def __init__(self, *, chunk_size_tokens: int, chunk_overlap_tokens: int) -> None:
        if chunk_overlap_tokens >= chunk_size_tokens:
            raise ValueError("overlap must be < size")
        self._size = chunk_size_tokens
        self._overlap = chunk_overlap_tokens

    def chunk(self, markdown: str, *, page_offsets: list[int] | None = None) -> list[Chunk]:
        raw = markdown.splitlines(keepends=True)
        bare = markdown.splitlines()
        n = len(raw)
        if n == 0:
            return []

        line_offset = [0] * n
        pos = 0
        for idx, line in enumerate(raw):
            line_offset[idx] = pos
            pos += len(line)

        table_ranges = table_line_ranges(bare)
        is_table_line = [False] * n
        for start, end in table_ranges:
            for k in range(start, end):
                is_table_line[k] = True

        line_path = self._line_section_paths(bare, is_table_line)

        chunks: list[Chunk] = []
        index = 0
        pending: list[_Word] = []  # accumulated prose words for the current run
        table_starts = {start: (end, ordinal) for ordinal, (start, end) in enumerate(table_ranges)}

        i = 0
        while i < n:
            if i in table_starts:
                # Close the current prose run, then emit the table as its own block.
                chunks, index = self._flush_prose(pending, markdown, page_offsets, chunks, index)
                pending = []
                end, table_index = table_starts[i]
                chunks, index = self._emit_table(
                    bare,
                    line_offset,
                    raw,
                    i,
                    end,
                    line_path[i],
                    page_offsets,
                    chunks,
                    index,
                    table_index,
                )
                i = end
                continue

            stripped = bare[i].strip()
            if not _HEADING_RE.match(stripped):  # heading lines contribute no words
                for match in _WORD_RE.finditer(bare[i]):
                    pending.append(
                        _Word(
                            text=match.group(0),
                            start=line_offset[i] + match.start(),
                            end=line_offset[i] + match.end(),
                            section_path=line_path[i],
                        )
                    )
            i += 1

        chunks, index = self._flush_prose(pending, markdown, page_offsets, chunks, index)
        return chunks

    @staticmethod
    def _line_section_paths(bare: list[str], is_table_line: list[bool]) -> list[tuple[str, ...]]:
        section_stack: list[tuple[int, str]] = []
        paths: list[tuple[str, ...]] = []
        for idx, line in enumerate(bare):
            stripped = line.strip()
            heading = _HEADING_RE.match(stripped)
            if heading and not is_table_line[idx]:
                level = len(heading.group(1))
                title = heading.group(2).strip()
                while section_stack and section_stack[-1][0] >= level:
                    section_stack.pop()
                section_stack.append((level, title))
            paths.append(tuple(title for _level, title in section_stack))
        return paths

    def _flush_prose(
        self,
        words: list[_Word],
        markdown: str,
        page_offsets: list[int] | None,
        chunks: list[Chunk],
        index: int,
    ) -> tuple[list[Chunk], int]:
        if not words:
            return chunks, index
        step = max(1, self._size - self._overlap)
        start = 0
        n = len(words)
        while start < n:
            window = words[start : start + self._size]
            text = markdown[window[0].start : window[-1].end].strip()
            page_from, page_to = self._page_range(window[0].start, window[-1].end, page_offsets)
            chunks.append(
                Chunk(
                    chunk_index=index,
                    text=text,
                    section_path=list(window[0].section_path),
                    page_from=page_from,
                    page_to=page_to,
                    token_count=len(window),
                )
            )
            index += 1
            if start + self._size >= n:
                break
            start += step
        return chunks, index

    def _emit_table(
        self,
        bare: list[str],
        line_offset: list[int],
        raw: list[str],
        start: int,
        end: int,
        section_path: tuple[str, ...],
        page_offsets: list[int] | None,
        chunks: list[Chunk],
        index: int,
        table_index: int,
    ) -> tuple[list[Chunk], int]:
        char_from = line_offset[start]
        char_to = line_offset[end - 1] + len(raw[end - 1])
        page_from, page_to = self._page_range(char_from, char_to, page_offsets)

        header, separator = bare[start], bare[start + 1]
        data = list(range(start + 2, end))
        header_tokens = len(_WORD_RE.findall(header))
        full_tokens = sum(len(_WORD_RE.findall(bare[k])) for k in range(start, end))

        # Small enough to keep the whole table together; otherwise split by data rows,
        # repeating the header (and separator) so every part keeps its column names.
        if full_tokens <= self._size or not data:
            groups = [data]
        else:
            groups = self._row_groups(bare, data, header_tokens)

        for group in groups:
            lines = [header, separator] + [bare[k] for k in group] if group else [header, separator]
            text = "\n".join(lines).strip()
            chunks.append(
                Chunk(
                    chunk_index=index,
                    text=text,
                    section_path=list(section_path),
                    page_from=page_from,
                    page_to=page_to,
                    token_count=len(_WORD_RE.findall(text)),
                    table_index=table_index,
                )
            )
            index += 1
        return chunks, index

    def _row_groups(self, bare: list[str], data: list[int], header_tokens: int) -> list[list[int]]:
        budget = max(1, self._size - header_tokens)
        groups: list[list[int]] = []
        current: list[int] = []
        used = 0
        for k in data:
            row_tokens = len(_WORD_RE.findall(bare[k]))
            if current and used + row_tokens > budget:
                groups.append(current)
                current = []
                used = 0
            current.append(k)
            used += row_tokens
        if current:
            groups.append(current)
        return groups

    @staticmethod
    def _page_range(
        start_offset: int, end_offset: int, page_offsets: list[int] | None
    ) -> tuple[int | None, int | None]:
        if not page_offsets:
            return None, None

        def page_for(offset: int) -> int:
            for page_num, boundary in enumerate(page_offsets, start=1):
                if offset < boundary:
                    return page_num
            return len(page_offsets)

        return page_for(start_offset), page_for(end_offset)
