"""Search returns the whole table for a table hit (search-improvement S2, Tier 4.1)."""

from __future__ import annotations

from app.search.service import SearchService


class _Settings:
    search_max_chunk_chars = 4000
    search_return_full_table = True


def _service(settings: _Settings) -> SearchService:
    return SearchService(store=None, vectors=None, embedding=None, settings=settings)


def test_table_hit_returns_the_whole_table() -> None:
    settings = _Settings()
    payload = {"text": "row 45: 562 500", "table_markdown": "| h |\n|---|\n| 45 | 562 500 |"}
    assert _service(settings)._result_text(payload) == payload["table_markdown"]


def test_non_table_hit_returns_the_chunk_text() -> None:
    settings = _Settings()
    assert _service(settings)._result_text({"text": "just prose"}) == "just prose"


def test_flag_off_returns_only_the_row_text() -> None:
    settings = _Settings()
    settings.search_return_full_table = False
    payload = {"text": "row 45: 562 500", "table_markdown": "WHOLE TABLE"}
    assert _service(settings)._result_text(payload) == "row 45: 562 500"


def test_full_table_is_length_bounded() -> None:
    settings = _Settings()
    settings.search_max_chunk_chars = 10
    payload = {"text": "row", "table_markdown": "X" * 100}
    assert _service(settings)._result_text(payload) == "X" * 10
