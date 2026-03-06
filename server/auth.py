"""Bearer token authentication for the TT-Train mock server."""

from __future__ import annotations

import os

from fastapi import Header, HTTPException, status


# If set, only this key is accepted. Otherwise any non-empty Bearer token works.
_REQUIRED_KEY = os.environ.get("TT_TRAIN_SERVER_API_KEY", "")


def verify_auth(authorization: str | None = Header(default=None)) -> str:
    """FastAPI dependency — validates Bearer token, returns the key."""
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": {"type": "authentication_error", "message": "Missing Authorization header"}},
        )

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": {"type": "authentication_error", "message": "Invalid Authorization header format"}},
        )

    if _REQUIRED_KEY and token.strip() != _REQUIRED_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": {"type": "authentication_error", "message": "Invalid API key"}},
        )

    return token.strip()


def error_404(resource: str, id: str) -> HTTPException:
    return HTTPException(
        status_code=404,
        detail={"error": {"type": "not_found_error", "message": f"{resource} not found", "code": f"{resource.lower()}_not_found", "id": id}},
    )


def error_400(message: str, code: str = "invalid_request") -> HTTPException:
    return HTTPException(
        status_code=400,
        detail={"error": {"type": "invalid_request_error", "message": message, "code": code}},
    )


def error_409(message: str, code: str = "conflict") -> HTTPException:
    return HTTPException(
        status_code=409,
        detail={"error": {"type": "conflict_error", "message": message, "code": code}},
    )
