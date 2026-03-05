"""Reward function management for RL training jobs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, IO

from tt_train.http import HTTPClient
from tt_train.types import RewardFunction, RewardTestResult, PaginatedList


class Rewards:
    """Upload and manage reward functions for RL training."""

    def __init__(self, http: HTTPClient):
        self._http = http

    def create(
        self,
        file: str | Path | IO,
        *,
        name: str | None = None,
        runtime: str = "python3.11",
        requirements: str | None = None,
    ) -> RewardFunction:
        """
        Upload a reward function.

        The file must export a `score` function::

            def score(prompt: str, completion: str, metadata: dict | None = None) -> float:
                ...

        Args:
            file: Python file path or file-like object.
            name: Human-readable name.
            runtime: Python runtime version.
            requirements: pip requirements (newline-separated).

        Returns:
            RewardFunction object (status "validating" initially).
        """
        fields: dict[str, str] = {"runtime": runtime}
        if name:
            fields["name"] = name
        if requirements:
            fields["requirements"] = requirements

        data = self._http.upload("/rewards", file=file, fields=fields)
        return RewardFunction.model_validate(data)

    def get(self, reward_id: str) -> RewardFunction:
        """Get a reward function by ID."""
        data = self._http.get(f"/rewards/{reward_id}")
        return RewardFunction.model_validate(data)

    def list(self, *, limit: int = 20, after: str | None = None) -> PaginatedList:
        """List reward functions."""
        data = self._http.get(
            "/rewards",
            params={"limit": limit, "after": after},
        )
        result = PaginatedList.model_validate(data)
        result.data = [RewardFunction.model_validate(r) for r in result.data]
        return result

    def delete(self, reward_id: str) -> dict[str, Any]:
        """Delete a reward function."""
        return self._http.delete(f"/rewards/{reward_id}")

    def test(
        self,
        reward_id: str,
        examples: list[dict[str, str]],
    ) -> RewardTestResult:
        """
        Test a reward function on sample inputs.

        Args:
            reward_id: Reward function ID.
            examples: List of {"prompt": ..., "completion": ...} dicts.

        Returns:
            RewardTestResult with scores and any errors.
        """
        data = self._http.post(
            f"/rewards/{reward_id}/test",
            json_body={"examples": examples},
        )
        return RewardTestResult.model_validate(data)

    def wait_until_ready(
        self,
        reward_id: str,
        *,
        poll_interval: float = 2.0,
        timeout: float = 120.0,
    ) -> RewardFunction:
        """Poll until reward function validation completes."""
        import time

        start = time.monotonic()
        while True:
            rf = self.get(reward_id)
            if rf.status == "ready":
                return rf
            if rf.status == "invalid":
                from tt_train.errors import InvalidRequestError
                raise InvalidRequestError(
                    f"Reward function {reward_id} failed validation",
                    code="reward_validation_failed",
                )
            if time.monotonic() - start > timeout:
                raise TimeoutError(f"Reward function {reward_id} not ready after {timeout}s")
            time.sleep(poll_interval)
