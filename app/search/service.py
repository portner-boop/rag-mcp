from __future__ import annotations

import asyncio
import time

import structlog

from app.search.fusion import FusedHit, reciprocal_rank_fusion
from app.search.normalize import normalize_query
from app.search.ports import SearchEmbedding, SearchStore, VectorSearch
from app.shared.contracts.embedding import RerankDocument
from app.shared.contracts.mcp import (
    SearchKnowledgeInput,
    SearchKnowledgeOutput,
    SearchMeta,
    SearchResult,
)
from app.shared.errors import ErrorCode, UpstreamError, ValidationError
from app.shared.ids import new_uuid
from app.storage.embedding import EmbeddingValidationError
from app.storage.qdrant import VectorHit

log = structlog.get_logger("search")

_EMBED_ERRORS = (UpstreamError, EmbeddingValidationError)


class SearchService:
    def __init__(
        self,
        *,
        store: SearchStore,
        vectors: VectorSearch,
        embedding: SearchEmbedding,
        settings,
        expander=None,
    ) -> None:
        self._store = store
        self._vectors = vectors
        self._embedding = embedding
        self._settings = settings
        self._expander = expander

    async def search_knowledge(self, payload: SearchKnowledgeInput) -> SearchKnowledgeOutput:
        from app.observability import metrics

        try:
            async with asyncio.timeout(self._settings.mcp_tool_timeout_seconds):
                return await self._run(payload)
        except TimeoutError as exc:
            metrics.search_requests_total.labels(status="timeout").inc()
            raise UpstreamError(
                "Search timed out", code=ErrorCode.QDRANT_TIMEOUT, retryable=True
            ) from exc

    async def _run(self, payload: SearchKnowledgeInput) -> SearchKnowledgeOutput:
        from app.observability import metrics

        started = time.perf_counter()
        query_id = new_uuid()

        normalized = normalize_query(payload.query)
        if not normalized:
            raise ValidationError("Query is empty after normalization", code=ErrorCode.EMPTY_QUERY)

        active = await self._store.get_active_index_config()

        if payload.document_ids:
            for document_id in payload.document_ids:
                if not await self._store.document_exists(document_id):
                    raise ValidationError(
                        "Unknown document id in filter", details={"document_id": document_id}
                    )
        excluded = await self._store.excluded_document_ids()

        dense_vec, sparse_vec = await self._representations(normalized)

        common = dict(
            limit=payload.max_candidates,
            index_version=active.version,
            document_ids=payload.document_ids,
            document_version=payload.filters.document_version,
            created_from=payload.filters.created_from,
            created_to=payload.filters.created_to,
            exclude_document_ids=excluded or None,
        )

        dense_hits, sparse_hits = await self._retrieve(dense_vec, sparse_vec, common)
        ranked_lists = [hits for hits in (dense_hits, sparse_hits) if hits]
        fused = reciprocal_rank_fusion(ranked_lists, k=self._settings.search_rrf_k)
        candidates = fused[: payload.max_candidates]

        final, reranked = await self._maybe_rerank(normalized, candidates, limit=payload.limit)

        expanded = False
        if self._should_expand(final, reranked):
            extra_lists = await self._expansion_lists(normalized, common)
            if extra_lists:
                expanded = True
                fused = reciprocal_rank_fusion(
                    ranked_lists + extra_lists, k=self._settings.search_rrf_k
                )
                candidates = fused[: payload.max_candidates]
                final, reranked = await self._maybe_rerank(
                    normalized, candidates, limit=payload.limit
                )

        results = self._to_results(final[: payload.limit], default_version=active.version)
        duration_ms = int((time.perf_counter() - started) * 1000)

        metrics.search_requests_total.labels(status="ok").inc()
        metrics.search_duration_seconds.observe(duration_ms / 1000)
        metrics.search_results_count.observe(len(results))
        if not results:
            metrics.search_empty_results_total.inc()

        log.info(
            "search.completed",
            query_id=query_id,
            dense=len(dense_hits),
            sparse=len(sparse_hits),
            reranked=reranked,
            expanded=expanded,
            results=len(results),
            duration_ms=duration_ms,
        )
        return SearchKnowledgeOutput(
            query_id=query_id,
            results=results,
            search_meta=SearchMeta(
                dense_candidates=len(dense_hits),
                sparse_candidates=len(sparse_hits),
                reranked=reranked,
                expanded=expanded,
                duration_ms=duration_ms,
            ),
        )

    async def _representations(
        self, query: str
    ) -> tuple[list[float] | None, tuple[list[int], list[float]] | None]:
        dense_vec: list[float] | None = None
        sparse_vec: tuple[list[int], list[float]] | None = None
        try:
            dense_vec = (await self._embedding.dense([self._dense_query(query)]))[0]
        except _EMBED_ERRORS:
            log.warning("search.dense_embedding_unavailable")
        try:
            sv = (await self._embedding.sparse([query]))[0]
            sparse_vec = (sv.indices, sv.values) if sv.indices else None
        except _EMBED_ERRORS:
            log.warning("search.sparse_embedding_unavailable")

        if dense_vec is None and sparse_vec is None:
            raise UpstreamError(
                "Query embeddings unavailable", code=ErrorCode.EMBEDDING_TIMEOUT, retryable=True
            )
        if sparse_vec is None and not self._settings.allow_dense_only_fallback:
            raise UpstreamError(
                "Sparse query embedding unavailable and dense-only fallback is disabled",
                code=ErrorCode.EMBEDDING_TIMEOUT,
                retryable=True,
            )
        if dense_vec is None and not self._settings.allow_sparse_only_fallback:
            raise UpstreamError(
                "Dense query embedding unavailable",
                code=ErrorCode.EMBEDDING_TIMEOUT,
                retryable=True,
            )
        return dense_vec, sparse_vec

    def _dense_query(self, query: str) -> str:
        instruction = getattr(self._settings, "embedding_query_instruction", "")
        return f"{instruction}{query}" if instruction else query

    async def _retrieve(
        self, dense_vec, sparse_vec, common
    ) -> tuple[list[VectorHit], list[VectorHit]]:
        from app.observability import metrics

        dense_hits: list[VectorHit] = []
        sparse_hits: list[VectorHit] = []
        if dense_vec is not None:
            t0 = time.perf_counter()
            dense_hits = await self._vectors.dense_search(vector=dense_vec, **common)
            metrics.dense_search_duration_seconds.observe(time.perf_counter() - t0)
        if sparse_vec is not None:
            t0 = time.perf_counter()
            sparse_hits = await self._vectors.sparse_search(
                indices=sparse_vec[0], values=sparse_vec[1], **common
            )
            metrics.sparse_search_duration_seconds.observe(time.perf_counter() - t0)
        return dense_hits, sparse_hits

    def _should_expand(self, final: list[FusedHit], reranked: bool) -> bool:
        if self._expander is None or not getattr(self._settings, "enable_query_expansion", False):
            return False
        if not final:
            return True
        if not reranked:
            return False
        threshold = getattr(self._settings, "query_expansion_min_rerank_score", 0.3)
        return final[0].score < threshold

    async def _expansion_lists(self, query: str, common) -> list[list[VectorHit]]:
        variants = await self._expand_queries(query)
        if not variants:
            return []
        gathered = await asyncio.gather(
            *(self._retrieve_for_query(v, common) for v in variants), return_exceptions=True
        )
        lists: list[list[VectorHit]] = []
        for result in gathered:
            if isinstance(result, tuple):
                dense_hits, sparse_hits = result
                if dense_hits:
                    lists.append(dense_hits)
                if sparse_hits:
                    lists.append(sparse_hits)
        return lists

    async def _expand_queries(self, query: str) -> list[str]:
        try:
            variants = await self._expander.expand(
                query,
                num_variants=getattr(self._settings, "query_expansion_max_variants", 3),
                hyde=getattr(self._settings, "enable_hyde", False),
            )
        except Exception:  # noqa: BLE001 - expansion is best-effort; never fail the search
            log.warning("search.expansion_unavailable")
            return []
        return [v for v in (variants or []) if v and v.strip()]

    async def _retrieve_for_query(
        self, query: str, common
    ) -> tuple[list[VectorHit], list[VectorHit]]:
        dense_vec, sparse_vec = await self._soft_representations(query)
        return await self._retrieve(dense_vec, sparse_vec, common)

    async def _soft_representations(self, query: str):
        dense_vec = None
        sparse_vec = None
        try:
            dense_vec = (await self._embedding.dense([self._dense_query(query)]))[0]
        except _EMBED_ERRORS:
            pass
        try:
            sv = (await self._embedding.sparse([query]))[0]
            sparse_vec = (sv.indices, sv.values) if sv.indices else None
        except _EMBED_ERRORS:
            pass
        return dense_vec, sparse_vec

    async def _maybe_rerank(
        self, query: str, candidates: list[FusedHit], *, limit: int
    ) -> tuple[list[FusedHit], bool]:
        from app.observability import metrics

        if not candidates or not self._settings.enable_reranker:
            return candidates, False
        docs = [
            RerankDocument(id=c.chunk_id, text=self._bound_text(c.payload.get("text", "")))
            for c in candidates
        ]
        try:
            t0 = time.perf_counter()
            resp = await self._embedding.rerank(query, docs, top_n=limit)
            metrics.rerank_duration_seconds.observe(time.perf_counter() - t0)
        except _EMBED_ERRORS:
            metrics.rerank_fallback_total.inc()
            log.warning("search.rerank_fallback")
            return candidates, False
        scores = {r.id: r.score for r in resp.results}
        by_id = {c.chunk_id: c for c in candidates}
        ordered = sorted(
            (cid for cid in scores if cid in by_id),
            key=lambda cid: (-scores[cid], cid),
        )
        reranked = [
            FusedHit(chunk_id=cid, score=scores[cid], payload=by_id[cid].payload) for cid in ordered
        ]
        return reranked, True

    def _to_results(self, hits: list[FusedHit], *, default_version: int) -> list[SearchResult]:
        results: list[SearchResult] = []
        for rank, hit in enumerate(hits, start=1):
            p = hit.payload
            results.append(
                SearchResult(
                    source_id=f"S{rank}",
                    chunk_id=hit.chunk_id,
                    document_id=p.get("document_id", ""),
                    filename=p.get("filename", ""),
                    text=self._result_text(p),
                    page_from=p.get("page_from"),
                    page_to=p.get("page_to"),
                    section_path=p.get("section_path", []),
                    score=float(hit.score),
                    index_version=p.get("index_version", default_version),
                )
            )
        return results

    def _result_text(self, payload: dict) -> str:
        text = payload.get("text", "")
        if getattr(self._settings, "search_return_full_table", True):
            table_markdown = payload.get("table_markdown")
            if table_markdown:
                text = table_markdown
        return self._bound_text(text)

    def _bound_text(self, text: str) -> str:
        limit = self._settings.search_max_chunk_chars
        return text if len(text) <= limit else text[:limit]
