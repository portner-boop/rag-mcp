# Domain Knowledge MCP

Autonomous, per-domain knowledge MCP stack — one deployable code base run separately for
each business domain (`hr`, `warehouse`, `finance`, …), each with its own PostgreSQL,
Qdrant collection, S3 bucket, RabbitMQ namespace, credentials and service tokens.

Implements slices **D01** (core, isolation, storage foundation) and **D02** (internal
corpus ingestion and indexing) of [`docs/domain-mcp/slices`](docs/domain-mcp/slices/README.md),
against [`docs/domain-mcp/tech-spec.md`](docs/domain-mcp/tech-spec.md) `1.3.0`.

The chat orchestrator sees exactly one tool — `search_knowledge`. Ingestion and corpus
maintenance live on a separate operational identity/network path and are never exposed to
the LLM (invariant 2).

## Architecture

Single `app/` package (bpmn-mcp "portable playbook" layout), FastMCP transport, Dynaconf
config. The server and worker are two **modes of the same app**, not separate packages.

```
app/
├── main.py               # entry: `rag-mcp {serve|worker|provision}`
├── server.py             # serve mode: chat MCP (FastMCP) + operational API + health
├── worker.py             # worker mode: ingestion queue consumer + run loop
├── container.py          # composition root + process-wide container accessor
├── config.py             # Dynaconf(settings.toml + RAG_MCP_ env) validated by Pydantic
├── logging.py            # structured JSON logs + trace correlation + secret redaction
├── provisioning.py       # seed active index config + Qdrant collection
├── tools/                # THE MCP SURFACE — one folder per tool (vertical slice)
│   ├── __init__.py       #   public_mcp; imports each slice to register it
│   ├── _common.py        #   shared tool-layer glue (container accessor, field annotations)
│   └── search/           #   search_knowledge slice
│       ├── tool.py       #     @tool — one-line delegate
│       └── service.py    #     orchestration → search engine
├── shared/               # cross-cutting primitives (no slice imports another)
│   ├── envelope.py       #   ok()/fail() + tool_wrapper (→ rag error contract)
│   ├── errors.py         #   DomainError + error codes (spec §13/§15)
│   ├── models.py enums.py ids.py trace.py time.py
│   └── contracts/        #   frozen MCP / queue / embedding Pydantic schemas
├── session/              # identity + credentials: auth.py (token verifier), identity.py
├── storage/              # external IO backends (named after the backend)
│   ├── base.py keys.py   #   protocol + pure key/path policy (security boundary)
│   ├── s3.py qdrant.py embedding.py
│   ├── embedding_openai.py  #   dense + rerank over an OpenAI/Cohere-shaped gateway
│   ├── bm25.py           #   lexical sparse vectors (Qdrant applies IDF)
│   ├── postgres/         #   engine, models, repositories
│   └── queue/            #   topology, publisher, connection, outbox_relay
├── ingestion/            # ingestion engine (worker pipeline)
│   ├── pipeline.py ports.py store_sql.py vector_qdrant.py consumer.py chunker.py
│   └── parser/           #   sandboxed parser → canonical Markdown
├── search/               # D03 hybrid retrieval: service, normalize, fusion, ports, store_sql
├── deletion/             # D04 deletion engine: pipeline, store_sql, consumer, ports
├── reindex/              # D04 reindex engine: pipeline (build→verify→cutover), store_sql, consumer
├── queueing/             # generic BaseConsumer (retry/DLQ/inbox) for deletion+reindex
├── worker_support/       # recovery loop + shared chunk/point helpers
├── operational/          # internal control plane (FastAPI): api, lifecycle, dlq, documents…
├── observability/        # prometheus metrics
└── testing/              # deterministic fakes + fake Embedding API
settings.toml             # committed non-secret tuning (Dynaconf)
Dockerfile / Dockerfile.embedding / docker-compose.yaml / .dockerignore
migrations/               # Alembic baseline schema
```

## Retrieval stack

```
query
 ├── dense   Qwen3 Embedding 8B (OpenRouter)  → top 50
 └── sparse  BM25, weights computed here, IDF by Qdrant → top 50
                          │
                     RRF fusion + dedup
                          │
              Qwen3 Reranker 4B (OpenRouter)
                          │
                    cited chunks → agent
```

Two knobs in `settings.toml` choose the backends:

| Setting | Values | Meaning |
|---|---|---|
| `embedding_provider` | `openai` (default) / `custom` | dense + rerank over an OpenAI/Cohere-shaped gateway, or over this stack's own Embedding API |
| `sparse_provider` | `bm25` (default) / `api` | lexical vectors computed in-process, or served by a sparse embedding model |

With the defaults the lexical branch needs no service at all: `POST /embeddings` and
`POST /rerank` go to the gateway, while BM25 term weights are computed in
[bm25.py](app/storage/bm25.py) and Qdrant supplies the IDF factor through the sparse
index `modifier: idf` — its built-in BM25 scoring. Point it at OpenRouter with two lines
of `.env`:

```bash
RAG_MCP_EMBEDDING_API_URL=https://openrouter.ai/api/v1
RAG_MCP_EMBEDDING_API_TOKEN=sk-or-...
```

`embedding_dense_dimension` is sent as `dimensions` on every embedding call, so the
vector always matches the collection. Changing it — or the dense model — invalidates
every stored vector, so startup refuses to run against an active index config that no
longer matches. In production that is a reindex (D04); in development:

```bash
uv run python scripts/reset_index.py --yes    # drop collection + corpus, re-provision
```

## Configuration

Dynaconf loads `settings.toml` (committed tuning) then overrides with `RAG_MCP_*` env and
`.env`. The merged values are validated by a Pydantic `Settings` model that keeps the
fail-fast startup and the **domain isolation guard** — a bucket/queue/collection name not
scoped to `DOMAIN_ID`, or a missing required value, still stops startup. Identity,
connection strings and secrets come from env (see `.env.example`).

## Run — Docker Compose (full stack)

```bash
# 1. token hashes (only the hash is stored, never the token)
python -c "from app.session.auth import hash_token; print(hash_token('chat-dev'))"
python -c "from app.session.auth import hash_token; print(hash_token('ops-dev'))"
# put them in .env as RAG_MCP_CHAT_SERVICE_TOKEN_HASH / RAG_MCP_OPS_SERVICE_TOKEN_HASH

# 2. bring up Postgres, Qdrant, MinIO(+bucket), RabbitMQ, fake-embedding, migrate, server, worker
docker compose --profile app up -d --build
```

Without `--profile app` the same file brings up the backing services only, which is what
you want when the server and worker run from your console (see below).

- Chat MCP: `POST http://localhost:8080/mcp` with `Authorization: Bearer chat-dev`
  (only `search_knowledge`). Readiness: `GET /health/ready`.
- Operational API: `http://localhost:8090` with `Authorization: Bearer ops-dev`
  (`/internal/documents/*`, `/internal/ingestion/*`, `/internal/downloads/*`),
  `/health/operational`, `/metrics`.

## Run — local (against the compose datastores)

```bash
uv sync
cp .env.example .env          # fill token hashes
uv run alembic upgrade head   # RAG_MCP_POSTGRES_URL is read by migrations/env.py
uv run rag-mcp provision      # seed index config v1 + Qdrant collection
uv run fake-embedding-api     # deterministic dev embeddings on :8000
uv run rag-mcp serve          # chat MCP :8080 + operational API :8090
uv run rag-mcp worker         # ingestion consumer
```

Compose publishes Postgres on host port **5433** (`RAG_MCP_POSTGRES_PORT` overrides it), so
a chat orchestrator with its own Postgres on 5432 can run side by side.

Put a document into the corpus through the operational identity — the chat surface only
searches:

```bash
uv run python scripts/ingest_local.py ./handbook.md    # prepare-upload → PUT → start → poll
```

## Tests (D05) & smoke checks

The pytest suite runs with no infrastructure (in-memory fakes):

```bash
uv run pytest -q                 # unit + property + contract + integration + security
uv run ruff check app scripts tests migrations
```

Layout (`tests/`): **unit** (state machine, chunking, fusion, retry backoff, compensation),
**property** (Hypothesis: stable IDs, ingestion idempotency, fusion + transition invariants),
**contract** (MCP schema, Embedding API, queue envelope, ops-interface auth), **integration**
(the 8 fake-backed E2E scenarios + consumer dispatch: handler-twice, cross-domain, retry, DLQ),
**security** (isolation guard, chat/ops identity separation, prompt-injection-as-data, log
redaction). Real-infra E2E and load/soak live in `tests/integration/test_real_infra.py`,
skipped unless `RUN_INTEGRATION=1` (they need Testcontainers/Docker).

Standalone smoke scripts (also fake-backed):

```bash
uv run python scripts/ingest_local.py FILE # push a local file through the ops ingest path
uv run python scripts/reset_index.py --yes # dev reset: drop collection + corpus, provision
uv run python scripts/smoke_config.py      # fail-fast config + domain isolation guard
uv run python scripts/smoke_ingestion.py   # pipeline: READY + idempotency + retrieval + failures
uv run python scripts/smoke_search.py      # D03: hybrid fuse+rerank, fallbacks, version/status exclusion
uv run python scripts/smoke_lifecycle.py   # D04: delete idempotency, reindex cutover, cancel
```

## D01–D04 checklist → where

| Item | Implementation |
|---|---|
| Server/worker start with valid config; mismatch stops startup | [config.py](app/config.py) isolation guard |
| Chat token cannot call operational interface | [session/auth.py](app/session/auth.py), split MCP vs ops apps |
| Chat surface exposes only `search_knowledge` | [tools/__init__.py](app/tools/__init__.py) |
| Ingestion reaches `READY` only after verified points; repeated event → one effect | [ingestion/pipeline.py](app/ingestion/pipeline.py) |
| **D03** original query → ranked hybrid chunks; dense/sparse/rerank + fallbacks | [search/service.py](app/search/service.py), [fusion.py](app/search/fusion.py) |
| **D03** active index version enforced; non-ready/deleting excluded | [storage/qdrant.py](app/storage/qdrant.py) `_search_filter` |
| **D03** per-call timeout / bounded results | [search/service.py](app/search/service.py) `asyncio.timeout` |
| **D04** delete idempotent; excludes from search immediately | [operational/lifecycle.py](app/operational/lifecycle.py), [deletion/pipeline.py](app/deletion/pipeline.py) |
| **D04** reindex builds target, verifies, atomic cutover; failure keeps old active | [reindex/pipeline.py](app/reindex/pipeline.py), [reindex/store_sql.py](app/reindex/store_sql.py) |
| **D04** cancel between stages; stale upload/job/lease recovery | [worker_support/recovery.py](app/worker_support/recovery.py), pipeline cancel checks |
| **D04** DLQ inspect/redrive under ops identity | [operational/dlq.py](app/operational/dlq.py) |
| **D04** ops S3/queue outage does not hide healthy search | separated readiness in [container.py](app/container.py) |

Operational procedures (deploy/rollback, backup/restore, rotation, egress, alerts):
[docs/domain-mcp/RUNBOOK.md](docs/domain-mcp/RUNBOOK.md).

## Status

Slices **D01–D05** are implemented. D05 ships the full fake-backed test matrix
(unit/property/contract/integration/security) covering the required E2E scenarios; the
real-infrastructure (Testcontainers) and load/soak variants are provided as gated stubs
(`RUN_INTEGRATION=1`) since they need a Docker daemon.
