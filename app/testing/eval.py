from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from app.storage.bm25 import tokenize
from app.worker_support.chunk_embed import TABLE_FACT_INDEX_BASE, chunk_markdown, embed_texts

_CHUNK_SIZE = 512
_CHUNK_OVERLAP = 64


@dataclass
class Unit:
    chunk_index: int
    kind: str
    indexed_text: str
    display_text: str
    filename: str
    section_path: tuple[str, ...]


@dataclass
class EvalCase:
    id: str
    query: str
    must_contain: list[str]
    note: str = ""


@dataclass
class CaseResult:
    case_id: str
    hit_rank: int | None
    hit_kind: str | None
    reciprocal_rank: float


@dataclass
class EvalReport:
    k: int
    cases: list[CaseResult] = field(default_factory=list)

    @property
    def recall_at_k(self) -> float:
        if not self.cases:
            return 0.0
        hits = sum(1 for c in self.cases if c.hit_rank is not None and c.hit_rank <= self.k)
        return hits / len(self.cases)

    @property
    def mrr(self) -> float:
        if not self.cases:
            return 0.0
        return sum(c.reciprocal_rank for c in self.cases) / len(self.cases)


def build_units(markdown: str, *, filename: str) -> list[Unit]:
    chunks, _version = chunk_markdown(
        markdown,
        page_offsets=None,
        chunk_size_tokens=_CHUNK_SIZE,
        chunk_overlap_tokens=_CHUNK_OVERLAP,
    )
    indexed = embed_texts(chunks, filename=filename)
    return [
        Unit(
            chunk_index=chunk.chunk_index,
            kind=_unit_kind(chunk),
            indexed_text=text,
            display_text=chunk.text,
            filename=filename,
            section_path=tuple(chunk.section_path),
        )
        for chunk, text in zip(chunks, indexed, strict=True)
    ]


def _unit_kind(chunk) -> str:
    if chunk.chunk_index >= TABLE_FACT_INDEX_BASE:
        return "fact"
    return "table" if chunk.text.lstrip().startswith("|") else "prose"


class Bm25Ranker:
    def __init__(self, units: list[Unit], *, k1: float = 1.2, b: float = 0.75) -> None:
        self._k1 = k1
        self._b = b
        self._docs = [tokenize(u.indexed_text) for u in units]
        self._n = len(self._docs)
        total = sum(len(d) for d in self._docs)
        self._avgdl = (total / self._n) if self._n else 1.0
        df: Counter[str] = Counter()
        for doc in self._docs:
            df.update(set(doc))
        self._idf = {
            term: math.log(1.0 + (self._n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()
        }

    def rank(self, query: str) -> list[int]:
        q_terms = set(tokenize(query))
        scored: list[tuple[float, int]] = []
        for i, doc in enumerate(self._docs):
            if not doc:
                scored.append((0.0, i))
                continue
            tf = Counter(doc)
            dl = len(doc)
            norm = self._k1 * (1.0 - self._b + self._b * dl / self._avgdl)
            score = 0.0
            for term in q_terms:
                f = tf.get(term, 0)
                if f:
                    score += self._idf.get(term, 0.0) * (f * (self._k1 + 1.0)) / (f + norm)
            scored.append((score, i))
        scored.sort(key=lambda s: (-s[0], s[1]))
        return [i for _score, i in scored]


def _relevant(unit: Unit, must_contain: list[str]) -> bool:
    return all(needle in unit.display_text for needle in must_contain)


def evaluate(cases: list[EvalCase], units: list[Unit], *, k: int = 10) -> EvalReport:
    ranker = Bm25Ranker(units)
    report = EvalReport(k=k)
    for case in cases:
        order = ranker.rank(case.query)
        hit_rank: int | None = None
        hit_kind: str | None = None
        for rank, unit_idx in enumerate(order, start=1):
            if _relevant(units[unit_idx], case.must_contain):
                hit_rank = rank
                hit_kind = units[unit_idx].kind
                break
        report.cases.append(
            CaseResult(
                case_id=case.id,
                hit_rank=hit_rank,
                hit_kind=hit_kind,
                reciprocal_rank=(1.0 / hit_rank) if hit_rank else 0.0,
            )
        )
    return report


def load_gold(gold_path: Path, *, fixtures_dir: Path) -> tuple[list[Unit], list[EvalCase]]:
    data = json.loads(gold_path.read_text(encoding="utf-8"))
    units: list[Unit] = []
    for entry in data["corpus"]:
        markdown = (fixtures_dir / entry["fixture"]).read_text(encoding="utf-8")
        units.extend(build_units(markdown, filename=entry["filename"]))
    cases = [
        EvalCase(
            id=c["id"],
            query=c["query"],
            must_contain=c["must_contain"],
            note=c.get("note", ""),
        )
        for c in data["cases"]
    ]
    return units, cases


def format_report(report: EvalReport, *, units_count: int) -> str:
    lines = [
        f"Retrieval eval (BM25 lexical) — {len(report.cases)} case(s), "
        f"{units_count} units, k={report.k}",
        f"  recall@{report.k}: {report.recall_at_k:.3f}    MRR: {report.mrr:.3f}",
        "",
    ]
    for c in report.cases:
        status = "MISS" if c.hit_rank is None else f"rank {c.hit_rank} ({c.hit_kind})"
        within = "ok " if (c.hit_rank is not None and c.hit_rank <= report.k) else "OUT"
        lines.append(f"  [{within}] {c.case_id}: {status}")
    return "\n".join(lines)
