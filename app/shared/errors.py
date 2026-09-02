from __future__ import annotations

from enum import Enum
from typing import Any


class ErrorCode(str, Enum):
    INVALID_INPUT = "INVALID_INPUT"
    NOT_FOUND = "NOT_FOUND"
    UNSUPPORTED_FILE = "UNSUPPORTED_FILE"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    EMPTY_QUERY = "EMPTY_QUERY"
    CORRUPTED_FILE = "CORRUPTED_FILE"
    IDEMPOTENCY_CONFLICT = "IDEMPOTENCY_CONFLICT"
    INVALID_STATE = "INVALID_STATE"

    MISSING_CAPABILITY = "MISSING_CAPABILITY"
    INVALID_INDEX_CONFIG = "INVALID_INDEX_CONFIG"
    INVALID_DIMENSION = "INVALID_DIMENSION"
    INVALID_COLLECTION = "INVALID_COLLECTION"

    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"

    EMBEDDING_TIMEOUT = "EMBEDDING_TIMEOUT"
    STORAGE_TIMEOUT = "STORAGE_TIMEOUT"
    QDRANT_TIMEOUT = "QDRANT_TIMEOUT"
    QUEUE_TIMEOUT = "QUEUE_TIMEOUT"
    UPSTREAM_UNAVAILABLE = "UPSTREAM_UNAVAILABLE"

    CROSS_DOMAIN_ACCESS = "CROSS_DOMAIN_ACCESS"
    STORAGE_CORRUPTION = "STORAGE_CORRUPTION"
    MISSING_POINTS = "MISSING_POINTS"
    INTERNAL = "INTERNAL"


NON_RETRYABLE: frozenset[ErrorCode] = frozenset(
    {
        ErrorCode.INVALID_INPUT,
        ErrorCode.UNSUPPORTED_FILE,
        ErrorCode.FILE_TOO_LARGE,
        ErrorCode.EMPTY_QUERY,
        ErrorCode.CORRUPTED_FILE,
        ErrorCode.IDEMPOTENCY_CONFLICT,
        ErrorCode.INVALID_STATE,
        ErrorCode.MISSING_CAPABILITY,
        ErrorCode.INVALID_INDEX_CONFIG,
        ErrorCode.INVALID_DIMENSION,
        ErrorCode.INVALID_COLLECTION,
        ErrorCode.UNAUTHORIZED,
        ErrorCode.FORBIDDEN,
        ErrorCode.CROSS_DOMAIN_ACCESS,
    }
)


class DomainError(Exception):
    code: ErrorCode = ErrorCode.INTERNAL
    http_status: int = 500

    def __init__(
        self,
        message: str,
        *,
        code: ErrorCode | None = None,
        retryable: bool | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        if code is not None:
            self.code = code
        self.message = message
        self.details = details or {}
        if retryable is None:
            retryable = self.code not in NON_RETRYABLE
        self.retryable = retryable

    def to_contract(self, request_id: str | None = None) -> dict[str, Any]:
        return {
            "error": {
                "code": self.code.value,
                "message": self.message,
                "retryable": self.retryable,
                "request_id": request_id,
                "details": self.details,
            }
        }


class ValidationError(DomainError):
    code = ErrorCode.INVALID_INPUT
    http_status = 400


class NotFoundError(DomainError):
    code = ErrorCode.NOT_FOUND
    http_status = 404


class UnauthorizedError(DomainError):
    code = ErrorCode.UNAUTHORIZED
    http_status = 401


class ForbiddenError(DomainError):
    code = ErrorCode.FORBIDDEN
    http_status = 403


class IdempotencyConflictError(DomainError):
    code = ErrorCode.IDEMPOTENCY_CONFLICT
    http_status = 409


class InvalidStateError(DomainError):
    code = ErrorCode.INVALID_STATE
    http_status = 409


class CrossDomainError(DomainError):
    code = ErrorCode.CROSS_DOMAIN_ACCESS
    http_status = 403


class ConfigurationError(DomainError):
    code = ErrorCode.INVALID_INDEX_CONFIG
    http_status = 500


class UpstreamError(DomainError):
    code = ErrorCode.UPSTREAM_UNAVAILABLE
    http_status = 503
