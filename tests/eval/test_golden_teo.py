"""Golden retrieval case for the search-improvement slice S1 (Tier 1.1).

Reference failure (see ``docs/search-improvement-plan.md``): the query
«стоимость эксплуатации при 45 процедурах у заказчика» must resolve to ``562 500 руб.``,
which lives in a single table cell (Таблица 17, row ``45``, column «Стоимость
эксплуатации»). The word-window chunker buried that value in a bibliography-dominated
chunk and stripped its column header, so the value was retrieved without meaning — or not
at all.

The fixture ``fixtures/teo.md`` is the real Docling Markdown of
``docs/ТЭО Планировочные решения.docx`` (committed so the test needs no ML models). These
assertions lock in the linearizer's guarantee: the answer value is emitted as a single
self-contained fact that binds row × column, uniquely across the whole document.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.ingestion.table_linearize import linearize_tables
from app.storage.bm25 import tokenize

FIXTURE = Path(__file__).parent / "fixtures" / "teo.md"

# The answer the system must be able to surface.
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
    """The 45-procedures row binds the answer value to its column and caption."""
    fact = _answer_fact(facts)
    assert ANSWER_VALUE in fact.text
    assert ANSWER_COLUMN in fact.text
    # The value is bound to *its* column, not an adjacent one.
    assert f"{ANSWER_COLUMN}, руб.: {ANSWER_VALUE}" in fact.text
    # Caption context travels with the row so the fact is interpretable on its own.
    assert "Таблица 17" in fact.text


def test_answer_value_is_uniquely_bound_across_the_document(facts):
    """Only the 45-procedures row ties «Стоимость эксплуатации» to «562 500».

    This is what lets retrieval disambiguate the answer from strong distractors that also
    mention «стоимость эксплуатации» and «45 процедур» (e.g. the risk-table prose "при
    частоте ниже 45 процедур ... стоимость эксплуатации подлежит снижению"), which do NOT
    carry the value.
    """
    bindings = [f for f in facts if ANSWER_VALUE in f.text and ANSWER_COLUMN in f.text]
    assert len(bindings) == 1
    assert bindings[0].cells[0] == "45"


def test_answer_fact_carries_the_discriminating_lexical_terms(facts):
    """BM25 tokens of the fact include the query's key terms co-located in one unit.

    ``стоимость``/``процедур`` appear here in document form; after Tier 2.1 lemmatization
    they also match the query's inflected ``стоимости``/``процедурах``. Co-location in a
    short fact (vs. a diluted 512-word chunk) is what makes the unit rank for the query.
    """
    fact = _answer_fact(facts)
    tokens = set(tokenize(fact.text))
    for term in ("стоимость", "эксплуатации", "процедур", "заказчика"):
        assert tokenize(term)[0] in tokens, f"missing discriminating term: {term}"
    for number in ("45", "562", "500"):
        assert number in tokens, f"missing number: {number}"


def test_frequency_table_section_path_is_correct(facts):
    """Linearization tracks headings per table, fixing the raw chunker's mislabel.

    The raw word-window chunk carrying this table was tagged 'ПРИЛОЖЕНИЕ В' while its
    content is in 'ПРИЛОЖЕНИЕ Г'; the per-table section path is accurate.
    """
    fact = _answer_fact(facts)
    assert fact.section_path and fact.section_path[-1] == "ПРИЛОЖЕНИЕ Г"


def test_stemmed_query_matches_previously_blocked_terms(facts):
    """Tier 2.1: the user's inflected query now matches the answer fact's surface forms.

    Before stemming, «стоимости»≠«стоимость» and «процедурах»≠«процедур», so the sparse
    branch fired on neither of the two most specific query terms. Both now match.
    """
    fact = _answer_fact(facts)
    shared = set(tokenize(USER_QUERY)) & set(tokenize(fact.text))
    assert tokenize("стоимость")[0] in shared  # previously blocked
    assert tokenize("процедур")[0] in shared  # previously blocked
    assert tokenize("эксплуатация")[0] in shared  # already matched, still does
    assert "45" in shared


@pytest.mark.asyncio
async def test_ingestion_indexes_the_answer_fact():
    """End-to-end (fakes): the full pipeline indexes the answer as a table-fact point.

    Runs parse → chunk+linearize → embed → upsert and asserts exactly one indexed point
    binds the value to its column, and that it is a table fact (reserved index range).
    """
    from app.testing.harness import build_ingestion_pipeline, ingestion_setup
    from app.worker_support.chunk_embed import TABLE_FACT_INDEX_BASE

    setup = ingestion_setup(
        text=FIXTURE.read_text(encoding="utf-8"), filename="ТЭО Планировочные решения.docx"
    )
    result = await build_ingestion_pipeline(setup).run(setup.event)
    assert result.status == "completed"

    # Exactly one *table fact* (reserved index range) binds the value to its column. The raw
    # table chunk (index below the base) also carries the value — that is intentional context.
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

    # Tier 4.1: the fact carries a stable table id and the WHOLE table (every row, not just
    # row 45), so a table hit can be returned to the LLM with header + neighbouring rows.
    assert payload.get("table_id")
    table_markdown = payload.get("table_markdown") or ""
    for opex in ("375 000", "562 500", "750 000", "1 000 000", "1 250 000"):
        assert opex in table_markdown, f"whole table missing row value {opex}"
