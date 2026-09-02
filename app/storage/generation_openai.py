"""Query expander over an OpenAI-shaped chat gateway (search-improvement S2, Tier 3.1).

One ``POST {base}/chat/completions`` turns a user query into a few retrieval-friendly
paraphrases and, optionally, a short hypothetical answer passage (HyDE) — extra query
strings whose embeddings pull in relevant chunks the literal query missed. This runs only
on the selective low-confidence path, so its cost is not paid on the common case.

The model output is untrusted text: we read it as newline-separated query strings, never as
instructions. A malformed or failed response degrades to "no expansion" rather than an error
that breaks search.
"""

from __future__ import annotations

import httpx

from app.shared.errors import ErrorCode, UpstreamError

_SYSTEM = (
    "You rewrite a search query to improve retrieval. Output ONLY query lines, one per line, "
    "no numbering, no commentary. Preserve the language of the query."
)


def _user_prompt(query: str, *, num_variants: int, hyde: bool) -> str:
    parts = [
        f"Original query:\n{query}\n",
        f"Write {num_variants} alternative phrasings that a relevant document might match "
        f"(synonyms, expansions of abbreviations, different word order).",
    ]
    if hyde:
        parts.append(
            "Then, on a final line, write one short factual sentence that would plausibly "
            "appear in a document answering the query (a hypothetical answer)."
        )
    return "\n".join(parts)


class OpenAICompatibleExpander:
    def __init__(
        self,
        *,
        base_url: str,
        token: str,
        model: str,
        timeout: float = 15.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._headers = {"Authorization": f"Bearer {token}"}
        self._transport = transport
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> OpenAICompatibleExpander:
        self._http()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url, headers=self._headers, transport=self._transport
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def expand(self, query: str, *, num_variants: int, hyde: bool) -> list[str]:
        user = _user_prompt(query, num_variants=num_variants, hyde=hyde)
        body = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": user},
            ],
            "temperature": 0.3,
            "max_tokens": 300,
        }
        try:
            resp = await self._http().post("/chat/completions", json=body, timeout=self._timeout)
            resp.raise_for_status()
            data = resp.json()
        except httpx.TimeoutException as exc:
            raise UpstreamError(
                "Query expansion timed out", code=ErrorCode.EMBEDDING_TIMEOUT, retryable=True
            ) from exc
        except httpx.HTTPError as exc:
            raise UpstreamError("Query expansion request failed") from exc

        content = _first_message_content(data)
        limit = num_variants + (1 if hyde else 0)
        return _parse_lines(content, query, limit=limit)


def _first_message_content(data: dict) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    return content if isinstance(content, str) else ""


def _parse_lines(content: str, original: str, *, limit: int) -> list[str]:
    """Newline-separated query strings, de-listed, de-duplicated, minus the original."""
    seen: dict[str, None] = {}
    original_norm = original.strip().casefold()
    for raw in content.splitlines():
        line = raw.strip().lstrip("-•*0123456789. \t").strip()
        if len(line) < 2 or line.casefold() == original_norm:
            continue
        seen.setdefault(line, None)
    return list(seen)[:limit]
