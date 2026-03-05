"""Job management — black-box fine-tuning."""

from __future__ import annotations

import time
from typing import Any, Generator

from tt_train.http import HTTPClient
from tt_train.types import Job, JobEvent, Checkpoint, JobEstimate, PaginatedList


class Jobs:
    """Create and manage black-box training jobs."""

    def __init__(self, http: HTTPClient):
        self._http = http

    def create(
        self,
        model: str,
        method: str,
        training_data: str,
        *,
        validation_data: str | None = None,
        config: dict[str, Any] | None = None,
        reward: dict[str, Any] | None = None,
        hardware: dict[str, Any] | None = None,
        integrations: list[dict[str, Any]] | None = None,
        webhooks: dict[str, str] | None = None,
        name: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> Job:
        """
        Create a training job.

        Args:
            model: Base model path, e.g. "tt://catalog/llama-3.2-8b".
            method: "sft", "dpo", "rl", or "pretrain".
            training_data: Dataset ID.
            validation_data: Optional validation dataset ID.
            config: Training configuration (epochs, lr, lora, etc.).
            reward: Reward config for RL jobs.
            hardware: Hardware preferences.
            integrations: W&B or other integrations.
            webhooks: Callback URLs for job events.
            name: Human-readable name.
            metadata: Arbitrary key-value pairs.

        Returns:
            Job object with status "created".
        """
        body: dict[str, Any] = {
            "model": model,
            "method": method,
            "training_data": training_data,
        }
        if validation_data is not None:
            body["validation_data"] = validation_data
        if config is not None:
            body["config"] = config
        if reward is not None:
            body["reward"] = reward
        if hardware is not None:
            body["hardware"] = hardware
        if integrations is not None:
            body["integrations"] = integrations
        if webhooks is not None:
            body["webhooks"] = webhooks
        if name is not None:
            body["name"] = name
        if metadata is not None:
            body["metadata"] = metadata

        data = self._http.post("/jobs", json_body=body)
        return Job.model_validate(data)

    def get(self, job_id: str) -> Job:
        """Get current job status and metrics."""
        data = self._http.get(f"/jobs/{job_id}")
        return Job.model_validate(data)

    def list(
        self,
        *,
        limit: int = 20,
        after: str | None = None,
        before: str | None = None,
        status: str | None = None,
        method: str | None = None,
    ) -> PaginatedList:
        """List jobs with optional filters."""
        data = self._http.get(
            "/jobs",
            params={
                "limit": limit,
                "after": after,
                "before": before,
                "status": status,
                "method": method,
            },
        )
        result = PaginatedList.model_validate(data)
        result.data = [Job.model_validate(j) for j in result.data]
        return result

    def cancel(self, job_id: str) -> Job:
        """Cancel a running job. Saves a checkpoint if possible."""
        data = self._http.post(f"/jobs/{job_id}/cancel")
        return Job.model_validate(data)

    def pause(self, job_id: str) -> Job:
        """Pause a running job. Saves checkpoint and releases hardware."""
        data = self._http.post(f"/jobs/{job_id}/pause")
        return Job.model_validate(data)

    def resume(
        self,
        job_id: str,
        *,
        hardware: dict[str, Any] | None = None,
    ) -> Job:
        """Resume a paused job. May be queued for hardware."""
        body = {}
        if hardware is not None:
            body["hardware"] = hardware
        data = self._http.post(f"/jobs/{job_id}/resume", json_body=body or None)
        return Job.model_validate(data)

    def list_checkpoints(self, job_id: str) -> list[Checkpoint]:
        """List all checkpoints for a job."""
        data = self._http.get(f"/jobs/{job_id}/checkpoints")
        return [Checkpoint.model_validate(c) for c in data.get("data", [])]

    def stream(
        self,
        job_id: str,
        *,
        after_step: int | None = None,
    ) -> Generator[JobEvent, None, None]:
        """
        Stream real-time events from a running job via SSE.

        Yields JobEvent objects with .event and .data attributes.
        """
        for raw in self._http.stream_sse(
            f"/jobs/{job_id}/events",
            params={"after_step": after_step},
        ):
            yield JobEvent(event=raw["event"], data=raw.get("data", {}))

    def estimate(
        self,
        model: str,
        method: str,
        training_data: str,
        *,
        config: dict[str, Any] | None = None,
        hardware: dict[str, Any] | None = None,
    ) -> JobEstimate:
        """Estimate cost and time for a job without creating it."""
        body: dict[str, Any] = {
            "model": model,
            "method": method,
            "training_data": training_data,
        }
        if config is not None:
            body["config"] = config
        if hardware is not None:
            body["hardware"] = hardware
        data = self._http.post("/jobs/estimate", json_body=body)
        return JobEstimate.model_validate(data)

    def wait(
        self,
        job_id: str,
        *,
        poll_interval: float = 10.0,
        timeout: float | None = None,
    ) -> Job:
        """
        Block until a job reaches a terminal state.

        Args:
            job_id: Job to wait for.
            poll_interval: Seconds between polls.
            timeout: Max seconds to wait (None = forever).

        Returns:
            Job in terminal state (completed, failed, cancelled).
        """
        terminal = {"completed", "failed", "cancelled"}
        start = time.monotonic()

        while True:
            job = self.get(job_id)
            if job.status in terminal:
                return job
            if timeout and (time.monotonic() - start > timeout):
                raise TimeoutError(
                    f"Job {job_id} still in state '{job.status}' after {timeout}s"
                )
            time.sleep(poll_interval)
