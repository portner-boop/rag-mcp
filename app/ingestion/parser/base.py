"""Parser interface and adapters (spec sections 3, 11).

A parser converts an original document to canonical Markdown. Adapters are selected by
content type. Document text is untrusted data, never executable instructions (spec 16);
adapters must not evaluate embedded macros/scripts. A production PDF/DOCX parser is
plugged in by configuration; the built-in registry ships text/markdown adapters and
rejects unsupported types deterministically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from app.shared.errors import DomainError, ErrorCode


class UnsupportedFileError(DomainError):
    code = ErrorCode.UNSUPPORTED_FILE
    http_status = 415


class EmptyMarkdownError(DomainError):
    code = ErrorCode.CORRUPTED_FILE
    http_status = 422


@dataclass
class ParsedDocument:
    markdown: str
    # Cumulative character offset at the end of each page, or None for page-less formats.
    page_offsets: list[int] | None = None
    parser_version: str = "text-1"
    meta: dict = field(default_factory=dict)


class Parser(Protocol):
    version: str

    def parse(self, data: bytes, *, filename: str) -> ParsedDocument: ...


class TextParser:
    """Plain text / Markdown pass-through parser."""

    version = "text-1"

    def parse(self, data: bytes, *, filename: str) -> ParsedDocument:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = data.decode("latin-1")
            except Exception as exc:  # noqa: BLE001
                raise EmptyMarkdownError("Could not decode document as text") from exc
        markdown = text.strip()
        if not markdown:
            raise EmptyMarkdownError("Parsed Markdown is empty")
        return ParsedDocument(markdown=markdown, page_offsets=None, parser_version=self.version)


class ParserRegistry:
    def __init__(self) -> None:
        self._by_type: dict[str, Parser] = {}

    def register(self, content_type: str, parser: Parser) -> None:
        self._by_type[content_type] = parser

    def get(self, content_type: str) -> Parser:
        parser = self._by_type.get(content_type)
        if parser is None:
            raise UnsupportedFileError(
                "No parser registered for content type",
                details={"content_type": content_type},
            )
        return parser


def default_registry() -> ParserRegistry:
    registry = ParserRegistry()
    text = TextParser()
    registry.register("text/plain", text)
    registry.register("text/markdown", text)
    registry.register("text/x-markdown", text)
    registry.register("text/csv", text)
    # Local import keeps adapters.py free to import from this module (no cycle).
    from app.ingestion.parser.adapters import (
        DOCX_CONTENT_TYPE,
        PDF_CONTENT_TYPE,
        PPTX_CONTENT_TYPE,
        XLSX_CONTENT_TYPE,
        DoclingParser,
        XlsxParser,
    )

    docling = DoclingParser()
    registry.register(PDF_CONTENT_TYPE, docling)
    registry.register(DOCX_CONTENT_TYPE, docling)
    registry.register(PPTX_CONTENT_TYPE, docling)
    registry.register(XLSX_CONTENT_TYPE, XlsxParser())
    return registry
