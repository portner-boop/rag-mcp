from __future__ import annotations

import pytest

from app.shared.enums import DocumentStatus
from app.shared.errors import InvalidStateError
from app.storage.postgres.models import Document
from app.storage.postgres.repositories import DocumentRepository

REPO = DocumentRepository(session=None)  # type: ignore[arg-type]


def _doc(status: DocumentStatus) -> Document:
    return Document(status=status.value, row_version=0)


LEGAL = [
    (DocumentStatus.UPLOADING, {DocumentStatus.UPLOADING}, DocumentStatus.UPLOADED),
    (
        DocumentStatus.UPLOADED,
        {DocumentStatus.UPLOADED, DocumentStatus.FAILED},
        DocumentStatus.QUEUED,
    ),
    (DocumentStatus.QUEUED, {DocumentStatus.QUEUED}, DocumentStatus.PROCESSING),
    (DocumentStatus.PROCESSING, {DocumentStatus.PROCESSING}, DocumentStatus.READY),
    (DocumentStatus.READY, {DocumentStatus.READY}, DocumentStatus.DELETING),
    (DocumentStatus.DELETING, {DocumentStatus.DELETING}, DocumentStatus.DELETED),
    (DocumentStatus.READY, {DocumentStatus.READY}, DocumentStatus.REINDEXING),
]

ILLEGAL = [
    (DocumentStatus.DELETED, {DocumentStatus.READY}, DocumentStatus.PROCESSING),
    (DocumentStatus.UPLOADING, {DocumentStatus.READY}, DocumentStatus.DELETING),
    (DocumentStatus.READY, {DocumentStatus.PROCESSING}, DocumentStatus.READY),
]


@pytest.mark.parametrize(("start", "allowed", "to"), LEGAL)
async def test_legal_transition(start, allowed, to) -> None:
    doc = _doc(start)
    await REPO.transition(doc, allowed_from=allowed, to=to)
    assert doc.status == to.value
    assert doc.row_version == 1


@pytest.mark.parametrize(("start", "allowed", "to"), ILLEGAL)
async def test_illegal_transition_raises(start, allowed, to) -> None:
    doc = _doc(start)
    with pytest.raises(InvalidStateError):
        await REPO.transition(doc, allowed_from=allowed, to=to)
    assert doc.status == start.value
