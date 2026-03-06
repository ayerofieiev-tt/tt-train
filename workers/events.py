"""
Platform event emitter — Training Service side.

The Platform defines what events it expects (see docs/tech/training-service-interface.md).
The Training Service emits them through this interface. What the Platform does with them
is outside the Training Service's concern.
"""
from __future__ import annotations

import logging
import uuid
from typing import Protocol, runtime_checkable

import httpx

logger = logging.getLogger(__name__)


@runtime_checkable
class EventEmitter(Protocol):
    def emit(self, event_type: str, *, status: str = "completed", **usage_fields) -> None:
        ...


class HttpEventEmitter:
    """Posts events to the Platform's metering endpoint."""

    def __init__(self, base_url: str, token: str, service_id: str, model: str, job_id: str):
        self._url = base_url.rstrip("/") + "/v1/events"
        self._headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        self._service_id = service_id
        self._model = model
        self._job_id = job_id

    def emit(self, event_type: str, *, status: str = "completed", **usage_fields) -> None:
        payload = {
            "event_id": str(uuid.uuid4()),
            "event_type": event_type,
            "service_id": self._service_id,
            "status": status,
            "usage": {"model": self._model, "job_id": self._job_id, **usage_fields},
        }
        try:
            r = httpx.post(self._url, json=payload, headers=self._headers, timeout=10.0)
            r.raise_for_status()
        except Exception as e:
            logger.warning("Failed to emit event %s: %s", event_type, e)


class NoopEventEmitter:
    """Used in standalone mode — events are logged but not sent anywhere."""

    def emit(self, event_type: str, *, status: str = "completed", **usage_fields) -> None:
        logger.debug("Event (noop): %s status=%s usage=%s", event_type, status, usage_fields)


def make_emitter(
    *,
    platform_base_url: str | None,
    worker_token: str | None,
    model: str,
    job_id: str,
    service_id: str = "tt-train",
) -> EventEmitter:
    """Return the appropriate emitter based on configuration."""
    if platform_base_url and worker_token:
        return HttpEventEmitter(
            base_url=platform_base_url,
            token=worker_token,
            service_id=service_id,
            model=model,
            job_id=job_id,
        )
    return NoopEventEmitter()
