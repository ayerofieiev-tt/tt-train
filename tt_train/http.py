"""Core HTTP client for the TT-Train API."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Generator, IO

import httpx

from tt_train.errors import raise_for_error, RateLimitError, TTTrainError


_DEFAULT_TIMEOUT = httpx.Timeout(60.0, connect=10.0)
_UPLOAD_TIMEOUT = httpx.Timeout(300.0, connect=10.0)
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 0.5


class HTTPClient:
    """Low-level HTTP client with auth, retries, and error handling."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.tt-train.dev/v1",
        organization: str | None = None,
        project: str | None = None,
        timeout: httpx.Timeout | None = None,
        max_retries: int = _MAX_RETRIES,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.organization = organization
        self.project = project
        self.max_retries = max_retries

        headers = {
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "tt-train-python/0.1.0",
        }
        if organization:
            headers["X-TT-Organization"] = organization
        if project:
            headers["X-TT-Project"] = project

        self._client = httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=timeout or _DEFAULT_TIMEOUT,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ----- Core request methods -----

    def request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """Make an authenticated request with automatic retries."""
        headers = {}
        if idempotency_key:
            headers["X-TT-Idempotency-Key"] = idempotency_key

        # Strip None values from params
        if params:
            params = {k: v for k, v in params.items() if v is not None}

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.request(
                    method,
                    path,
                    json=json_body,
                    params=params,
                    headers=headers,
                )
                return self._handle_response(response)

            except RateLimitError as e:
                last_error = e
                wait = e.retry_after or _RETRY_BASE_DELAY * (2 ** attempt)
                if attempt < self.max_retries:
                    time.sleep(wait)
                    continue
                raise

            except TTTrainError:
                raise  # Don't retry client errors

            except httpx.HTTPStatusError as e:
                if e.response.status_code in _RETRYABLE_STATUS and attempt < self.max_retries:
                    last_error = e
                    time.sleep(_RETRY_BASE_DELAY * (2 ** attempt))
                    continue
                raise

        raise last_error  # type: ignore[misc]

    def _handle_response(self, response: httpx.Response) -> dict[str, Any]:
        """Parse response, raise on errors."""
        if response.status_code >= 400:
            try:
                body = response.json()
            except (json.JSONDecodeError, ValueError):
                body = {"error": {"type": "internal_error", "message": response.text}}
            raise_for_error(response.status_code, body)

        if response.status_code == 204:
            return {}

        return response.json()

    # ----- Convenience methods -----

    def get(self, path: str, **kwargs) -> dict[str, Any]:
        return self.request("GET", path, **kwargs)

    def post(self, path: str, **kwargs) -> dict[str, Any]:
        return self.request("POST", path, **kwargs)

    def delete(self, path: str, **kwargs) -> dict[str, Any]:
        return self.request("DELETE", path, **kwargs)

    # ----- File upload -----

    def upload(
        self,
        path: str,
        *,
        file: str | Path | IO,
        fields: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Upload a file via multipart/form-data."""
        if isinstance(file, (str, Path)):
            file = open(file, "rb")
            should_close = True
        else:
            should_close = False

        try:
            files = {"file": file}
            data = fields or {}

            # For multipart uploads, we need to let httpx set the Content-Type
            # with the boundary. We temporarily remove our default JSON content type.
            response = self._client.post(
                path,
                files=files,
                data=data,
                timeout=_UPLOAD_TIMEOUT,
            )

            if response.status_code >= 400:
                try:
                    body = response.json()
                except (json.JSONDecodeError, ValueError):
                    body = {"error": {"type": "internal_error", "message": response.text}}
                raise_for_error(response.status_code, body)

            return response.json()
        finally:
            if should_close:
                file.close()  # type: ignore

    # ----- Server-Sent Events -----

    def stream_sse(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> Generator[dict[str, Any], None, None]:
        """Stream Server-Sent Events. Yields dicts with 'event' and 'data' keys."""
        if params:
            params = {k: v for k, v in params.items() if v is not None}

        with self._client.stream(
            "GET",
            path,
            params=params,
            headers={"Accept": "text/event-stream"},
            timeout=httpx.Timeout(None, connect=10.0),  # no read timeout for SSE
        ) as response:
            if response.status_code >= 400:
                response.read()
                try:
                    body = response.json()
                except (json.JSONDecodeError, ValueError):
                    body = {"error": {"type": "internal_error", "message": response.text}}
                raise_for_error(response.status_code, body)

            current_event = "message"
            current_data_lines: list[str] = []

            for line in response.iter_lines():
                if line.startswith("event:"):
                    current_event = line[6:].strip()

                elif line.startswith("data:"):
                    current_data_lines.append(line[5:].strip())

                elif line == "":
                    # End of event
                    if current_data_lines:
                        raw_data = "\n".join(current_data_lines)
                        if raw_data == "[DONE]":
                            return

                        try:
                            data = json.loads(raw_data)
                        except json.JSONDecodeError:
                            data = {"raw": raw_data}

                        yield {"event": current_event, "data": data}

                    current_event = "message"
                    current_data_lines = []
