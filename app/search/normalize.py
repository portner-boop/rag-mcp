"""Query normalization that must not change intent (spec section 12.2 step 1).

Conservative: Unicode NFC, strip control characters, trim and collapse whitespace. Casing
and word choice are left to the embedding model so intent is preserved.
"""

from __future__ import annotations

import re
import unicodedata

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WS = re.compile(r"\s+")


def normalize_query(query: str) -> str:
    text = unicodedata.normalize("NFC", query)
    text = _CONTROL.sub(" ", text)
    text = _WS.sub(" ", text).strip()
    return text
