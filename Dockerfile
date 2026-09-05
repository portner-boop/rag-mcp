FROM ghcr.io/astral-sh/uv:0.9.11 AS uv

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    UV_HTTP_TIMEOUT=600

WORKDIR /app

COPY --from=uv /uv /usr/local/bin/uv
COPY pyproject.toml uv.lock README.md ./
COPY app ./app
COPY migrations ./migrations
COPY alembic.ini settings.toml ./

RUN uv sync --locked --no-dev --no-editable \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin ragmcp

USER ragmcp

# Chat MCP (8080) and operational API (8090). The worker uses no ports.
EXPOSE 8080 8090

# Override in compose: `rag-mcp worker`, `rag-mcp provision`, or `alembic upgrade head`.
CMD ["rag-mcp", "serve"]
