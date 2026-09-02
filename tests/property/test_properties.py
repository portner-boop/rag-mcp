from __future__ import annotations

import asyncio

from hypothesis import given
from hypothesis import strategies as st

from app.ingestion.pipeline import PipelineResult
from app.search.fusion import reciprocal_rank_fusion
from app.shared.enums import DocumentStatus
from app.shared.errors import InvalidStateError
from app.shared.ids import stable_point_id
from app.storage.postgres.models import Document
from app.storage.postgres.repositories import DocumentRepository
from app.storage.qdrant import VectorHit
from app.testing.harness import build_ingestion_pipeline, ingestion_setup

_WORD = st.from_regex(r"[a-z]{1,8}", fullmatch=True)


@given(
    doc=st.uuids().map(str),
    dv=st.integers(1, 5),
    iv=st.integers(1, 5),
    ci=st.integers(0, 500),
)
def test_stable_point_id_is_deterministic(doc, dv, iv, ci) -> None:
    assert stable_point_id(doc, dv, iv, ci) == stable_point_id(doc, dv, iv, ci)


@given(doc=st.uuids().map(str), n=st.integers(1, 100), dv=st.integers(1, 3), iv=st.integers(1, 3))
def test_stable_point_ids_unique_per_chunk(doc, n, dv, iv) -> None:
    ids = [stable_point_id(doc, dv, iv, ci) for ci in range(n)]
    assert len(set(ids)) == n


@given(doc=st.uuids().map(str))
def test_stable_point_id_varies_with_version_axes(doc) -> None:
    assert stable_point_id(doc, 1, 1, 0) != stable_point_id(doc, 2, 1, 0)
    assert stable_point_id(doc, 1, 1, 0) != stable_point_id(doc, 1, 2, 0)


@given(
    dense=st.lists(st.tuples(st.text(min_size=1, max_size=4), st.floats(0, 1)), max_size=20),
    sparse=st.lists(st.tuples(st.text(min_size=1, max_size=4), st.floats(0, 1)), max_size=20),
)
def test_fusion_is_deterministic_and_dedupes(dense, sparse) -> None:
    d = [VectorHit(i, s, {}) for i, s in dense]
    s = [VectorHit(i, sc, {}) for i, sc in sparse]
    first = reciprocal_rank_fusion([d, s])
    assert first == reciprocal_rank_fusion([d, s])
    ids = [f.chunk_id for f in first]
    assert len(ids) == len(set(ids))
    scores = [f.score for f in first]
    assert scores == sorted(scores, reverse=True)


@given(words=st.lists(_WORD, min_size=3, max_size=60))
def test_ingestion_idempotent_no_double_effect(words) -> None:
    async def scenario() -> None:
        setup = ingestion_setup(text="# Doc\n\n" + " ".join(words))
        r1: PipelineResult = await build_ingestion_pipeline(setup).run(setup.event)
        assert r1.status == "completed"
        count1 = await setup.vectors.count_for_document(setup.document_id, index_version=1)
        r2 = await build_ingestion_pipeline(setup).run(setup.event)
        assert r2.status == "duplicate"
        count2 = await setup.vectors.count_for_document(setup.document_id, index_version=1)
        assert count1 == count2 == r1.chunk_count
        completed = [
            e for e in setup.store.outbox if e["event_type"] == "DocumentIngestionCompleted"
        ]
        assert len(completed) == 1

    asyncio.run(scenario())


_STATUSES = list(DocumentStatus)


@given(
    start=st.sampled_from(_STATUSES),
    allowed=st.lists(st.sampled_from(_STATUSES), min_size=1, max_size=4).map(set),
    to=st.sampled_from(_STATUSES),
)
def test_transition_respects_allowed_set(start, allowed, to) -> None:
    async def scenario() -> None:
        repo = DocumentRepository(session=None)  # type: ignore[arg-type]
        doc = Document(status=start.value, row_version=0)
        if start in allowed:
            await repo.transition(doc, allowed_from=allowed, to=to)
            assert doc.status == to.value
        else:
            try:
                await repo.transition(doc, allowed_from=allowed, to=to)
                raise AssertionError("expected InvalidStateError")
            except InvalidStateError:
                assert doc.status == start.value

    asyncio.run(scenario())
