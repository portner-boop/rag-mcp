from __future__ import annotations

import re
import secrets
from contextvars import ContextVar
from dataclasses import dataclass

_TRACEPARENT_RE = re.compile(
    r"^(?P<version>[0-9a-f]{2})-(?P<trace_id>[0-9a-f]{32})-"
    r"(?P<span_id>[0-9a-f]{16})-(?P<flags>[0-9a-f]{2})$"
)

_current: ContextVar[TraceContext | None] = ContextVar("trace_context", default=None)


@dataclass(frozen=True)
class TraceContext:
    trace_id: str
    span_id: str
    request_id: str
    sampled: bool = True

    def to_traceparent(self) -> str:
        flags = "01" if self.sampled else "00"
        return f"00-{self.trace_id}-{self.span_id}-{flags}"

    def child_span(self) -> TraceContext:
        return TraceContext(
            trace_id=self.trace_id,
            span_id=_random_hex(16),
            request_id=self.request_id,
            sampled=self.sampled,
        )


def _random_hex(chars: int) -> str:
    return secrets.token_hex(chars // 2)


def new_trace(request_id: str | None = None) -> TraceContext:
    return TraceContext(
        trace_id=_random_hex(32),
        span_id=_random_hex(16),
        request_id=request_id or _random_hex(16),
        sampled=True,
    )


def parse_traceparent(value: str | None, request_id: str | None = None) -> TraceContext:
    if value:
        match = _TRACEPARENT_RE.match(value.strip())
        if match:
            return TraceContext(
                trace_id=match["trace_id"],
                span_id=_random_hex(16),
                request_id=request_id or _random_hex(16),
                sampled=match["flags"].endswith("1"),
            )
    return new_trace(request_id)


def set_current(ctx: TraceContext | None) -> None:
    _current.set(ctx)


def current() -> TraceContext | None:
    return _current.get()


def current_trace_id() -> str | None:
    ctx = _current.get()
    return ctx.trace_id if ctx else None
