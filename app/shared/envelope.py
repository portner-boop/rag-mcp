from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from typing import Any

import structlog

from app.shared.errors import DomainError, ErrorCode
from app.shared.models import Envelope
from app.shared.trace import current_trace_id

log = structlog.get_logger("tools")


def ok(
    result: Any = None, *, warnings: list[Any] | None = None, hints: list[str] | None = None
) -> dict:
    return Envelope(
        ok=True,
        result=result if result is not None else {},
        warnings=warnings or [],
        hints=hints or [],
    ).model_dump()


def fail(errors: Any, *, code: str = "TOOL_ERROR", hints: list[str] | None = None) -> dict:
    errs = errors if isinstance(errors, list) else [errors]
    return Envelope(ok=False, result={}, errors=errs, code=code, hints=hints or []).model_dump()


def _error_contract(exc: Exception) -> dict:
    if isinstance(exc, DomainError):
        return exc.to_contract(current_trace_id())
    log.exception("tool.internal_error")
    return DomainError("Internal error", code=ErrorCode.INTERNAL).to_contract(current_trace_id())


def tool_wrapper(fn: Callable[..., Any]) -> Callable[..., Any]:
    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def awrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await fn(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                return _error_contract(exc)

        return awrapper

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            return _error_contract(exc)

    return wrapper


safe_tool = tool_wrapper
