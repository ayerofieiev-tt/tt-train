"""Errors raised by the TT-Train SDK."""

from __future__ import annotations

from typing import Any


class TTTrainError(Exception):
    """Base class for all TT-Train errors."""

    def __init__(
        self,
        message: str,
        *,
        type: str | None = None,
        code: str | None = None,
        param: str | None = None,
        status_code: int | None = None,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.type = type
        self.code = code
        self.param = param
        self.status_code = status_code
        self.details = details or {}

    def __repr__(self) -> str:
        parts = [f"message={self.args[0]!r}"]
        if self.code:
            parts.append(f"code={self.code!r}")
        if self.param:
            parts.append(f"param={self.param!r}")
        if self.status_code:
            parts.append(f"status_code={self.status_code}")
        return f"{self.__class__.__name__}({', '.join(parts)})"


class AuthenticationError(TTTrainError):
    """Invalid or missing API key."""
    pass


class InvalidRequestError(TTTrainError):
    """Malformed request or invalid parameters."""
    pass


class NotFoundError(TTTrainError):
    """Resource not found."""
    pass


class ConflictError(TTTrainError):
    """Resource state conflict (e.g. cancelling a completed job)."""
    pass


class RateLimitError(TTTrainError):
    """Rate limit exceeded."""

    def __init__(self, message: str, *, retry_after: float | None = None, **kwargs):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class HardwareUnavailableError(TTTrainError):
    """Requested hardware is not available."""
    pass


class SessionExpiredError(TTTrainError):
    """Session timed out due to inactivity."""

    def __init__(self, message: str, *, last_checkpoint: str | None = None, **kwargs):
        super().__init__(message, **kwargs)
        self.last_checkpoint = last_checkpoint


class QuotaExceededError(TTTrainError):
    """Organization quota exceeded."""
    pass


class InternalError(TTTrainError):
    """Server-side error — safe to retry."""
    pass


class OverloadedError(TTTrainError):
    """Service temporarily at capacity."""
    pass


# ---------------------------------------------------------------------------
# Mapping from API error types to exception classes
# ---------------------------------------------------------------------------

_ERROR_TYPE_MAP: dict[str, type[TTTrainError]] = {
    "authentication_error": AuthenticationError,
    "invalid_request_error": InvalidRequestError,
    "not_found_error": NotFoundError,
    "conflict_error": ConflictError,
    "rate_limit_error": RateLimitError,
    "quota_exceeded_error": QuotaExceededError,
    "hardware_unavailable_error": HardwareUnavailableError,
    "session_expired_error": SessionExpiredError,
    "internal_error": InternalError,
    "overloaded_error": OverloadedError,
}


def raise_for_error(status_code: int, body: dict[str, Any]) -> None:
    """Parse an error response body and raise the appropriate exception."""
    err = body.get("error", {})
    error_type = err.get("type", "internal_error")
    message = err.get("message", "Unknown error")
    code = err.get("code")
    param = err.get("param")
    details = err.get("details")

    cls = _ERROR_TYPE_MAP.get(error_type, TTTrainError)

    kwargs: dict[str, Any] = dict(
        type=error_type,
        code=code,
        param=param,
        status_code=status_code,
        details=details,
    )

    # RateLimitError extras
    if cls is RateLimitError:
        kwargs["retry_after"] = err.get("retry_after_seconds")

    # SessionExpiredError extras
    if cls is SessionExpiredError:
        kwargs["last_checkpoint"] = err.get("last_checkpoint")

    raise cls(message, **kwargs)
