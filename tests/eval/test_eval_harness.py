from __future__ import annotations

from pathlib import Path

import pytest

from app.testing.eval import evaluate, load_gold

_EVAL_DIR = Path(__file__).parent
_GOLD = _EVAL_DIR / "gold" / "teo.json"
_FIXTURES = _EVAL_DIR / "fixtures"

K = 10


@pytest.fixture(scope="module")
def report():
    units, cases = load_gold(_GOLD, fixtures_dir=_FIXTURES)
    assert units and cases
    return evaluate(cases, units, k=K)


def test_recall_at_k_is_perfect_on_the_gold_set(report):
    misses = [c.case_id for c in report.cases if c.hit_rank is None or c.hit_rank > K]
    assert not misses, f"cases missed at k={K}: {misses}"
    assert report.recall_at_k == 1.0


def test_mrr_stays_high(report):
    assert report.mrr >= 0.75


def test_reference_opex_case_is_top_ranked(report):
    case = next(c for c in report.cases if c.case_id == "teo-opex-45")
    assert case.hit_rank == 1
    assert case.hit_kind == "fact"
