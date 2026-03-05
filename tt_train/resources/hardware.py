"""Hardware catalog and availability."""

from __future__ import annotations

from tt_train.http import HTTPClient
from tt_train.types import HardwareCatalog


class Hardware:
    """Query hardware availability and pricing."""

    def __init__(self, http: HTTPClient):
        self._http = http

    def catalog(self) -> HardwareCatalog:
        """
        Get available hardware, pricing, and current queue estimates.

        Returns:
            HardwareCatalog with accelerator details and queue info.
        """
        data = self._http.get("/hardware/catalog")
        return HardwareCatalog.model_validate(data)
