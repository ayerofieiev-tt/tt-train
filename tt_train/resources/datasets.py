"""Dataset management."""

from __future__ import annotations

from pathlib import Path
from typing import Any, IO

from tt_train.http import HTTPClient
from tt_train.types import Dataset, PaginatedList


class Datasets:
    """Upload, list, and manage training datasets."""

    def __init__(self, http: HTTPClient):
        self._http = http

    def create(
        self,
        file: str | Path | IO,
        *,
        format: str,
        name: str | None = None,
        description: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> Dataset:
        """
        Upload a dataset.

        Args:
            file: Path to a JSONL file, or a file-like object.
            format: One of "chat", "completion", "preference", "reward".
            name: Human-readable name.
            description: Optional description.
            metadata: Arbitrary key-value pairs.

        Returns:
            Dataset object (status will be "processing" initially).
        """
        fields: dict[str, str] = {"format": format}
        if name:
            fields["name"] = name
        if description:
            fields["description"] = description
        if metadata:
            import json
            fields["metadata"] = json.dumps(metadata)

        data = self._http.upload("/datasets", file=file, fields=fields)
        return Dataset.model_validate(data)

    def get(self, dataset_id: str) -> Dataset:
        """Get a dataset by ID."""
        data = self._http.get(f"/datasets/{dataset_id}")
        return Dataset.model_validate(data)

    def list(
        self,
        *,
        limit: int = 20,
        after: str | None = None,
        before: str | None = None,
        format: str | None = None,
    ) -> PaginatedList:
        """List datasets."""
        data = self._http.get(
            "/datasets",
            params={"limit": limit, "after": after, "before": before, "format": format},
        )
        result = PaginatedList.model_validate(data)
        result.data = [Dataset.model_validate(d) for d in result.data]
        return result

    def delete(self, dataset_id: str) -> dict[str, Any]:
        """Delete a dataset. Fails if referenced by a running job."""
        return self._http.delete(f"/datasets/{dataset_id}")

    def wait_until_ready(
        self,
        dataset_id: str,
        *,
        poll_interval: float = 2.0,
        timeout: float = 300.0,
    ) -> Dataset:
        """Poll until dataset processing completes."""
        import time

        start = time.monotonic()
        while True:
            ds = self.get(dataset_id)
            if ds.status == "ready":
                return ds
            if ds.status not in ("processing", "uploading"):
                from tt_train.errors import InvalidRequestError
                raise InvalidRequestError(
                    f"Dataset {dataset_id} has status '{ds.status}'",
                    code="dataset_processing_failed",
                )
            if time.monotonic() - start > timeout:
                raise TimeoutError(f"Dataset {dataset_id} not ready after {timeout}s")
            time.sleep(poll_interval)
