from __future__ import annotations

from pathlib import Path

import pytest

from app.ingestion.table_linearize import linearize_tables
from app.storage.bm25 import tokenize

FIXTURE = Path(__file__).parent / "fixtures" / "teo.md"

ANSWER_VALUE = "562 500"
ANSWER_COLUMN = "Стоимость эксплуатации"
FREQ_TABLE_CAPTION = "частоты выполнения процедуры"
USER_QUERY = "назови цифру стоимости эксплуатации при 45 процедурах у заказчика ? ТЭО"


@pytest.fixture(scope="module")
def facts():
    return linearize_tables(FIXTURE.read_text(encoding="utf-8"))


def _answer_fact(facts):
    freq_rows = [f for f in facts if FREQ_TABLE_CAPTION in f.caption.lower()]
    assert freq_rows, "frequency-vs-cost table not found in the fixture"
    matches = [f for f in freq_rows if f.cells and f.cells[0] == "45"]
    assert len(matches) == 1, f"expected exactly one '45' row, got {len(matches)}"
    return matches[0]


def test_answer_row_is_a_self_contained_fact(facts):
    fact = _answer_fact(facts)
    assert ANSWER_VALUE in fact.text
    assert ANSWER_COLUMN in fact.text
    assert f"{ANSWER_COLUMN}, руб.: {ANSWER_VALUE}" in fact.text
    assert "Таблица 17" in fact.text


def test_answer_value_is_uniquely_bound_across_the_document(facts):
    bindings = [f for f in facts if ANSWER_VALUE in f.text and ANSWER_COLUMN in f.text]
    assert len(bindings) == 1
    assert bindings[0].cells[0] == "45"


def test_answer_fact_carries_the_discriminating_lexical_terms(facts):
    fact = _answer_fact(facts)
    tokens = set(tokenize(fact.text))
    for term in ("стоимость", "эксплуатации", "процедур", "заказчика"):
        assert tokenize(term)[0] in tokens, f"missing discriminating term: {term}"
    for number in ("45", "562", "500"):
        assert number in tokens, f"missing number: {number}"


def test_frequency_table_section_path_is_correct(facts):
    fact = _answer_fact(facts)
    assert fact.section_path and fact.section_path[-1] == "ПРИЛОЖЕНИЕ Г"


def test_stemmed_query_matches_previously_blocked_terms(facts):
    fact = _answer_fact(facts)
    shared = set(tokenize(USER_QUERY)) & set(tokenize(fact.text))
    assert tokenize("стоимость")[0] in shared
    assert tokenize("процедур")[0] in shared
    assert tokenize("эксплуатация")[0] in shared
    assert "45" in shared


@pytest.mark.asyncio
async def test_ingestion_indexes_the_answer_fact():
    from app.testing.harness import build_ingestion_pipeline, ingestion_setup
    from app.worker_support.chunk_embed import TABLE_FACT_INDEX_BASE

    setup = ingestion_setup(
        text=FIXTURE.read_text(encoding="utf-8"), filename="ТЭО Планировочные решения.docx"
    )
    result = await build_ingestion_pipeline(setup).run(setup.event)
    assert result.status == "completed"

    answer_facts = [
        p
        for p in setup.vectors.points.values()
        if p.payload["chunk_index"] >= TABLE_FACT_INDEX_BASE
        and ANSWER_VALUE in p.payload.get("text", "")
        and ANSWER_COLUMN in p.payload.get("text", "")
    ]
    assert len(answer_facts) == 1
    payload = answer_facts[0].payload
    assert f"{ANSWER_COLUMN}, руб.: {ANSWER_VALUE}" in payload["text"]

    assert payload.get("table_id")
    table_markdown = payload.get("table_markdown") or ""
    for opex in ("375 000", "562 500", "750 000", "1 000 000", "1 250 000"):
        assert opex in table_markdown, f"whole table missing row value {opex}"
