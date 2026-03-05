"""High-level TT-Train client."""

from __future__ import annotations

from tt_train.http import HTTPClient
from tt_train.resources.datasets import Datasets
from tt_train.resources.models import Models
from tt_train.resources.jobs import Jobs
from tt_train.resources.sessions import Sessions
from tt_train.resources.inference import Inference
from tt_train.resources.rewards import Rewards
from tt_train.resources.hardware import Hardware


class Client:
    """
    TT-Train API client.

    Usage::

        client = tt.Client(api_key="tt-...")
        job = client.jobs.create(
            model="tt://catalog/llama-3.2-8b",
            method="sft",
            training_data="ds_abc123",
        )

    Or configure via environment::

        # export TT_TRAIN_API_KEY=tt-...
        client = tt.Client()
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        organization: str | None = None,
        project: str | None = None,
        base_url: str = "https://api.tt-train.dev/v1",
        max_retries: int = 3,
    ):
        import os

        resolved_key = api_key or os.environ.get("TT_TRAIN_API_KEY")
        if not resolved_key:
            from tt_train.errors import AuthenticationError
            raise AuthenticationError(
                "No API key provided. Pass api_key= or set TT_TRAIN_API_KEY."
            )

        self._http = HTTPClient(
            api_key=resolved_key,
            base_url=base_url,
            organization=organization,
            project=project,
            max_retries=max_retries,
        )

        self.datasets = Datasets(self._http)
        self.models = Models(self._http)
        self.jobs = Jobs(self._http)
        self.sessions = Sessions(self._http)
        self.inference = Inference(self._http)
        self.rewards = Rewards(self._http)
        self.hardware = Hardware(self._http)

    def close(self) -> None:
        """Release underlying HTTP connections."""
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
