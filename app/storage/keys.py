from __future__ import annotations

import posixpath
import re

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_filename(filename: str) -> str:
    base = posixpath.basename(filename.replace("\\", "/")).strip()
    base = base.replace("..", "_")
    safe = _UNSAFE.sub("_", base).strip("._-")
    if not safe:
        safe = "file"
    return safe[:200]


def original_key(document_id: str, filename: str) -> str:
    return f"documents/{document_id}/original/{sanitize_filename(filename)}"


def markdown_key(document_id: str) -> str:
    return f"documents/{document_id}/parsed/document.md"
