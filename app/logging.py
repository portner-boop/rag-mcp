from __future__ import annotations

import logging
import re

import structlog

from app.shared.trace import current as current_trace

_TOKEN_RE = re.compile(r"(token=)[^&\s\"'>]+", re.IGNORECASE)


def _add_trace(_logger, _method, event_dict):  # noqa: ANN001
    ctx = current_trace()
    if ctx is not None:
        event_dict.setdefault("trace_id", ctx.trace_id)
        event_dict.setdefault("request_id", ctx.request_id)
    return event_dict


def _redact(_logger, _method, event_dict):  # noqa: ANN001
    event = event_dict.get("event")
    if isinstance(event, str):
        event_dict["event"] = _TOKEN_RE.sub(r"\1[REDACTED]", event)
    return event_dict


def setup_logging(
    level: str = "INFO",
    *,
    service: str = "domain-mcp",
    domain: str = "",
    environment: str = "local",
) -> None:
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(format="%(message)s", level=log_level)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            _add_trace,
            _redact,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        cache_logger_on_first_use=True,
    )
    structlog.contextvars.bind_contextvars(service=service, domain=domain, environment=environment)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
