from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

BackendState = Literal["pending", "running", "done", "failed", "cancelled"]


class ClusterBackend(ABC):
    # Subclasses set these to an int to cap local concurrency.
    # None means "no limit" (Slurm owns scheduling).
    max_concurrent_jobs: int | None = None
    max_concurrent_sessions: int | None = None

    @abstractmethod
    async def submit_job(
        self,
        *,
        job_id: str,
        script_path: str,
        args: list[str],
        nodes: int = 1,
        partition: str | None = None,
        account: str | None = None,
        env: dict[str, str] | None = None,
    ) -> str:
        """Submit a batch job. Returns backend job ID."""

    @abstractmethod
    async def submit_session(
        self,
        *,
        session_id: str,
        script_path: str,
        args: list[str],
        nodes: int = 1,
        partition: str | None = None,
        account: str | None = None,
        env: dict[str, str] | None = None,
    ) -> str:
        """Submit an interactive allocation. Returns backend job ID."""

    @abstractmethod
    async def cancel(self, backend_job_id: str) -> None:
        """Cancel a running job."""

    @abstractmethod
    async def get_state(self, backend_job_id: str) -> BackendState:
        """Get current state of a backend job."""
