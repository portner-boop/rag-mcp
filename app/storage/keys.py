"""Server-generated S3 object keys and filename sanitization (spec section 14).

Object keys are always server-generated and never accepted from the caller
(invariant/security section 16). Layout:

    <domain>-documents/
    └── documents/{document_id}/
        ├── original/{sanitized_original_filename}
        └── parsed/document.md
"""

from __future__ import annotations

import posixpath
import re

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_filename(filename: str) -> str:
    """Reduce a display filename to a safe key segment, preserving the extension."""
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
