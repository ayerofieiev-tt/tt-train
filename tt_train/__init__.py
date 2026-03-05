"""
TT-Train Python SDK
~~~~~~~~~~~~~~~~~~~

Fine-tuning and training on Tenstorrent hardware.

Quick start::

    import tt_train as tt

    tt.api_key = "tt-..."

    # Black-box fine-tuning
    job = tt.jobs.create(
        model="tt://catalog/llama-3.2-8b",
        method="sft",
        training_data="ds_abc123",
    )

    # Interactive session
    session = tt.sessions.create(
        model="tt://catalog/llama-3.2-8b",
        lora={"rank": 64},
        optimizer={"type": "adamw", "lr": 2e-5},
    )
    result = session.forward_backward(batch=data, loss="cross_entropy")
    session.step()
"""

from tt_train.client import Client
from tt_train.resources.datasets import Datasets
from tt_train.resources.models import Models
from tt_train.resources.jobs import Jobs
from tt_train.resources.sessions import Sessions, Session
from tt_train.resources.inference import Inference
from tt_train.resources.rewards import Rewards
from tt_train.resources.hardware import Hardware
from tt_train.errors import (
    TTTrainError,
    AuthenticationError,
    InvalidRequestError,
    NotFoundError,
    RateLimitError,
    HardwareUnavailableError,
    SessionExpiredError,
    QuotaExceededError,
    ConflictError,
)

__all__ = [
    "Client",
    "api_key",
    "organization",
    "project",
    "base_url",
    "datasets",
    "models",
    "jobs",
    "sessions",
    "inference",
    "rewards",
    "hardware",
    # Errors
    "TTTrainError",
    "AuthenticationError",
    "InvalidRequestError",
    "NotFoundError",
    "RateLimitError",
    "HardwareUnavailableError",
    "SessionExpiredError",
    "QuotaExceededError",
    "ConflictError",
]

# ---------------------------------------------------------------------------
# Module-level configuration (convenience API)
# ---------------------------------------------------------------------------
# Users can either configure the module-level defaults:
#     tt.api_key = "tt-..."
#     tt.jobs.create(...)
#
# Or create an explicit client:
#     client = tt.Client(api_key="tt-...")
#     client.jobs.create(...)
# ---------------------------------------------------------------------------

api_key: str | None = None
organization: str | None = None
project: str | None = None
base_url: str = "https://api.tt-train.dev/v1"


def _get_default_client() -> Client:
    """Lazily build a client from module-level config."""
    import tt_train as _mod

    if _mod.api_key is None:
        import os

        _mod.api_key = os.environ.get("TT_TRAIN_API_KEY")
    if _mod.api_key is None:
        raise AuthenticationError(
            "No API key provided. Set tt_train.api_key or the "
            "TT_TRAIN_API_KEY environment variable."
        )
    return Client(
        api_key=_mod.api_key,
        organization=_mod.organization,
        project=_mod.project,
        base_url=_mod.base_url,
    )


class _ModuleProxy:
    """Proxy that forwards attribute access to a lazily-created default client."""

    def __init__(self, resource_name: str):
        self._resource_name = resource_name

    def __getattr__(self, name: str):
        client = _get_default_client()
        resource = getattr(client, self._resource_name)
        return getattr(resource, name)


# Module-level resource accessors
datasets: Datasets = _ModuleProxy("datasets")  # type: ignore[assignment]
models: Models = _ModuleProxy("models")  # type: ignore[assignment]
jobs: Jobs = _ModuleProxy("jobs")  # type: ignore[assignment]
sessions: Sessions = _ModuleProxy("sessions")  # type: ignore[assignment]
inference: Inference = _ModuleProxy("inference")  # type: ignore[assignment]
rewards: Rewards = _ModuleProxy("rewards")  # type: ignore[assignment]
hardware: Hardware = _ModuleProxy("hardware")  # type: ignore[assignment]
