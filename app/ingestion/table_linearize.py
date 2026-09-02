from __future__ import annotations

import re
from dataclasses import dataclass, field

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_SEPARATOR_RE = re.compile(r"^\s*\|?[\s:|-]*-[\s:|-]*\|?\s*$")
_BOLD_RE = re.compile(r"(\*\*|__|`)")
_WS_RE = re.compile(r"\s+")


@dataclass
class TableFact:
    table_index: int
    row_index: int
    caption: str
    section_path: tuple[str, ...]
    headers: list[str] = field(default_factory=list)
    cells: list[str] = field(default_factory=list)
    text: str = ""
    table_markdown: str = ""


def table_line_ranges(lines: list[str]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    i = 0
    n = len(lines)
    while i < n:
        if _is_row(lines[i]) and i + 1 < n and _SEPARATOR_RE.match(lines[i + 1]):
            j = i + 2
            while j < n and _is_row(lines[j]) and not _SEPARATOR_RE.match(lines[j]):
                j += 1
            ranges.append((i, j))
            i = j
        else:
            i += 1
    return ranges


def linearize_tables(markdown: str) -> list[TableFact]:
    lines = markdown.splitlines()
    section_stack: list[tuple[int, str]] = []
    facts: list[TableFact] = []
    table_index = 0
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()

        heading = _HEADING_RE.match(stripped)
        if heading:
            level = len(heading.group(1))
            title = _strip_cell(heading.group(2))
            while section_stack and section_stack[-1][0] >= level:
                section_stack.pop()
            section_stack.append((level, title))
            i += 1
            continue

        if _is_row(line) and i + 1 < n and _SEPARATOR_RE.match(lines[i + 1]):
            headers = _split_row(line)
            caption = _find_caption(lines, i)
            section_path = tuple(title for _level, title in section_stack)
            end = i + 2
            while end < n and _is_row(lines[end]) and not _SEPARATOR_RE.match(lines[end]):
                end += 1
            table_markdown = _table_markdown(lines, i, end, caption)
            row_index = 0
            for k in range(i + 2, end):
                cells = _split_row(lines[k])
                if any(cells):
                    facts.append(
                        _build_fact(
                            table_index,
                            row_index,
                            caption,
                            section_path,
                            headers,
                            cells,
                            table_markdown,
                        )
                    )
                    row_index += 1
            table_index += 1
            i = end
            continue

        i += 1
    return facts


def _table_markdown(lines: list[str], start: int, end: int, caption: str) -> str:
    body = "\n".join(line.rstrip() for line in lines[start:end])
    return f"{caption}\n{body}" if caption else body


def _build_fact(
    table_index: int,
    row_index: int,
    caption: str,
    section_path: tuple[str, ...],
    headers: list[str],
    cells: list[str],
    table_markdown: str,
) -> TableFact:
    pairs: list[str] = []
    for idx, value in enumerate(cells):
        if not value:
            continue
        header = headers[idx] if idx < len(headers) else ""
        pairs.append(f"{header}: {value}" if header else value)
    body = "; ".join(pairs)
    text = f"{caption}. {body}." if caption else f"{body}."
    return TableFact(
        table_index=table_index,
        row_index=row_index,
        caption=caption,
        section_path=section_path,
        headers=headers,
        cells=cells,
        text=text,
        table_markdown=table_markdown,
    )


def _is_row(line: str) -> bool:
    return "|" in line and line.strip() != ""


def _split_row(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [_strip_cell(cell) for cell in s.split("|")]


def _strip_cell(cell: str) -> str:
    return _WS_RE.sub(" ", _BOLD_RE.sub("", cell)).strip()


def _find_caption(lines: list[str], header_idx: int) -> str:
    k = header_idx - 1
    while k >= 0:
        stripped = lines[k].strip()
        if not stripped:
            k -= 1
            continue
        if _HEADING_RE.match(stripped) or _is_row(lines[k]) or _SEPARATOR_RE.match(lines[k]):
            return ""
        return _strip_cell(stripped)
    return ""
