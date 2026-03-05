"""Session management — interactive training with low-level primitives."""

from __future__ import annotations

import time
from typing import Any

from tt_train.http import HTTPClient
from tt_train.types import (
    SessionInfo,
    SampleResult,
    ForwardBackwardResult,
    StepResult,
    Checkpoint,
    LogProbsResult,
    EvalResult,
    PaginatedList,
)


class Sessions:
    """Create and manage interactive training sessions."""

    def __init__(self, http: HTTPClient):
        self._http = http

    def create(
        self,
        model: str,
        *,
        lora: dict[str, Any] | None = None,
        optimizer: dict[str, Any] | None = None,
        lr_scheduler: dict[str, Any] | None = None,
        max_seq_len: int = 4096,
        max_grad_norm: float | None = 1.0,
        hardware: dict[str, Any] | None = None,
        idle_timeout_minutes: int = 30,
        name: str | None = None,
        metadata: dict[str, str] | None = None,
        wait: bool = True,
        wait_timeout: float = 300.0,
    ) -> Session:
        """
        Create an interactive training session.

        This allocates hardware and loads the model. By default, blocks
        until the session is ready.

        Args:
            model: Base model or checkpoint path.
            lora: LoRA config. None = full fine-tuning.
            optimizer: Optimizer config (type, lr, etc.).
            lr_scheduler: LR scheduler config.
            max_seq_len: Maximum sequence length.
            max_grad_norm: Default gradient clipping.
            hardware: Hardware preferences.
            idle_timeout_minutes: Auto-close after inactivity.
            name: Human-readable name.
            metadata: Arbitrary key-value pairs.
            wait: Block until session is ready.
            wait_timeout: Max seconds to wait for ready state.

        Returns:
            Session handle with training primitives.
        """
        body: dict[str, Any] = {"model": model, "max_seq_len": max_seq_len}

        if lora is not None:
            body["lora"] = lora
        if optimizer is not None:
            body["optimizer"] = optimizer
        if lr_scheduler is not None:
            body["lr_scheduler"] = lr_scheduler
        if max_grad_norm is not None:
            body["max_grad_norm"] = max_grad_norm
        if hardware is not None:
            body["hardware"] = hardware
        body["idle_timeout_minutes"] = idle_timeout_minutes
        if name is not None:
            body["name"] = name
        if metadata is not None:
            body["metadata"] = metadata

        data = self._http.post("/sessions", json_body=body)
        info = SessionInfo.model_validate(data)
        session = Session(self._http, info)

        if wait:
            session.wait_until_ready(timeout=wait_timeout)

        return session

    def get(self, session_id: str) -> Session:
        """Reconnect to an existing session."""
        data = self._http.get(f"/sessions/{session_id}")
        info = SessionInfo.model_validate(data)
        return Session(self._http, info)

    def list(
        self,
        *,
        limit: int = 20,
        after: str | None = None,
        status: str | None = None,
    ) -> PaginatedList:
        """List sessions."""
        data = self._http.get(
            "/sessions",
            params={"limit": limit, "after": after, "status": status},
        )
        result = PaginatedList.model_validate(data)
        result.data = [SessionInfo.model_validate(s) for s in result.data]
        return result


class Session:
    """
    A live training session with low-level primitives.

    This is the core object for interactive / agent-driven training.
    Each method maps to a single API call.

    Usage::

        session = tt.sessions.create(
            model="tt://catalog/llama-3.2-8b",
            lora={"rank": 64},
            optimizer={"type": "adamw", "lr": 2e-5},
        )

        for batch in dataloader:
            result = session.forward_backward(batch=batch, loss="cross_entropy")
            session.step()
            print(f"step {session.step_count}: loss={result.loss:.4f}")

        session.save(name="final")
        session.close()
    """

    def __init__(self, http: HTTPClient, info: SessionInfo):
        self._http = http
        self._info = info

    # ----- Properties -----

    @property
    def id(self) -> str:
        return self._info.id

    @property
    def model(self) -> str:
        return self._info.model

    @property
    def status(self) -> str:
        return self._info.status

    @property
    def step_count(self) -> int:
        return self._info.step_count

    @property
    def total_cost(self) -> str:
        return self._info.total_cost

    @property
    def last_checkpoint(self) -> str | None:
        return self._info.last_checkpoint

    @property
    def info(self) -> SessionInfo:
        return self._info

    # ----- Lifecycle -----

    def refresh(self) -> SessionInfo:
        """Fetch latest session state from the server."""
        data = self._http.get(f"/sessions/{self.id}")
        self._info = SessionInfo.model_validate(data)
        return self._info

    def wait_until_ready(self, *, timeout: float = 300.0, poll_interval: float = 2.0):
        """Block until the session is ready for use."""
        start = time.monotonic()
        while True:
            self.refresh()
            if self._info.status == "ready":
                return
            if self._info.status in ("expired", "closed", "failed"):
                from tt_train.errors import SessionExpiredError
                raise SessionExpiredError(
                    f"Session {self.id} entered state '{self._info.status}' "
                    f"while waiting for ready.",
                    last_checkpoint=self._info.last_checkpoint,
                )
            if time.monotonic() - start > timeout:
                raise TimeoutError(
                    f"Session {self.id} not ready after {timeout}s "
                    f"(current status: {self._info.status})"
                )
            time.sleep(poll_interval)

    def close(self) -> SessionInfo:
        """
        Close the session, releasing hardware.

        An auto-checkpoint is saved. Returns final session info.
        """
        data = self._http.delete(f"/sessions/{self.id}")
        self._info = SessionInfo.model_validate(data)
        return self._info

    # ----- Training Primitives -----

    def sample(
        self,
        prompts: list[dict | str],
        *,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        max_tokens: int = 256,
        n: int = 1,
        stop: list[str] | None = None,
        return_log_probs: bool = False,
        seed: int | None = None,
    ) -> SampleResult:
        """
        Generate completions from current model weights.

        Args:
            prompts: List of prompts. Each can be a messages list or a string.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.
            top_k: Top-k sampling (-1 to disable).
            max_tokens: Max tokens per completion.
            n: Number of completions per prompt.
            stop: Stop sequences.
            return_log_probs: Include per-token log probabilities.
            seed: Random seed for reproducibility.

        Returns:
            SampleResult with completions and usage info.
        """
        # Normalize prompts to message format
        normalized = []
        for p in prompts:
            if isinstance(p, str):
                normalized.append({"messages": [{"role": "user", "content": p}]})
            elif isinstance(p, list):
                normalized.append({"messages": p})
            elif isinstance(p, dict):
                normalized.append(p if "messages" in p else {"messages": [p]})
            else:
                normalized.append(p)

        body: dict[str, Any] = {
            "prompts": normalized,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "n": n,
            "return_log_probs": return_log_probs,
        }
        if top_k != -1:
            body["top_k"] = top_k
        if stop is not None:
            body["stop"] = stop
        if seed is not None:
            body["seed"] = seed

        data = self._http.post(f"/sessions/{self.id}/sample", json_body=body)
        return SampleResult.model_validate(data)

    def forward_backward(
        self,
        batch: list[dict[str, Any]],
        *,
        loss: str = "cross_entropy",
        loss_config: dict[str, Any] | None = None,
    ) -> ForwardBackwardResult:
        """
        Compute loss and gradients on a batch.

        Gradients accumulate until step() is called.

        Args:
            batch: Training examples. Format depends on loss type.
            loss: Loss function — "cross_entropy", "dpo", "reinforce", "grpo", "ppo".
            loss_config: Loss-specific parameters (e.g. {"beta": 0.1} for DPO).

        Returns:
            ForwardBackwardResult with loss, grad_norm, cost.
        """
        body: dict[str, Any] = {"batch": batch, "loss": loss}
        if loss_config is not None:
            body["loss_config"] = loss_config

        data = self._http.post(f"/sessions/{self.id}/forward_backward", json_body=body)
        result = ForwardBackwardResult.model_validate(data)

        # Update local step count if the server returns it
        self._info.step_count = max(self._info.step_count, result.session_id and 0)
        return result

    def step(self, *, max_grad_norm: float | None = None) -> StepResult:
        """
        Apply accumulated gradients via the optimizer.

        Clears the gradient buffer after update.

        Args:
            max_grad_norm: Override gradient clipping for this step.

        Returns:
            StepResult with step number, learning rate, grad norms.
        """
        body: dict[str, Any] = {}
        if max_grad_norm is not None:
            body["max_grad_norm"] = max_grad_norm

        data = self._http.post(
            f"/sessions/{self.id}/step",
            json_body=body if body else None,
        )
        result = StepResult.model_validate(data)
        self._info.step_count = result.step_number
        return result

    def save(
        self,
        *,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Checkpoint:
        """
        Save a checkpoint of the current model state.

        The checkpoint becomes a reusable model in your project.

        Args:
            name: Human-readable checkpoint name.
            metadata: Arbitrary metadata to attach.

        Returns:
            Checkpoint with model_path for later use.
        """
        body: dict[str, Any] = {}
        if name is not None:
            body["name"] = name
        if metadata is not None:
            body["metadata"] = metadata

        data = self._http.post(
            f"/sessions/{self.id}/save",
            json_body=body if body else None,
        )
        ckpt = Checkpoint.model_validate(data)
        self._info.last_checkpoint = ckpt.model_path
        return ckpt

    def log_probs(
        self,
        batch: list[dict[str, Any]],
    ) -> LogProbsResult:
        """
        Score completions without computing gradients.

        Useful for evaluation, computing reference policy log-probs for DPO,
        or reward model training.

        Args:
            batch: List of {"prompt": ..., "completion": ...} dicts.

        Returns:
            LogProbsResult with per-example and per-token scores.
        """
        data = self._http.post(
            f"/sessions/{self.id}/log_probs",
            json_body={"batch": batch},
        )
        return LogProbsResult.model_validate(data)

    def eval(
        self,
        data: str | list[dict[str, Any]],
        *,
        metrics: list[str] | None = None,
        max_examples: int | None = None,
        batch_size: int = 32,
    ) -> EvalResult:
        """
        Run evaluation (forward pass only).

        Args:
            data: Dataset ID (string) or inline examples (list of dicts).
            metrics: Metrics to compute — e.g. ["loss", "perplexity", "accuracy"].
            max_examples: Cap evaluation set size.
            batch_size: Batch size for eval.

        Returns:
            EvalResult with aggregate metrics.
        """
        body: dict[str, Any] = {}
        if isinstance(data, str):
            body["data"] = data
        else:
            body["data"] = {"inline": data}
        if metrics is not None:
            body["metrics"] = metrics
        if max_examples is not None:
            body["max_examples"] = max_examples
        body["batch_size"] = batch_size

        resp = self._http.post(f"/sessions/{self.id}/eval", json_body=body)
        return EvalResult.model_validate(resp)

    # ----- Context manager -----

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self._info.status not in ("closed", "expired"):
            self.close()

    def __repr__(self) -> str:
        return (
            f"Session(id={self.id!r}, model={self.model!r}, "
            f"status={self.status!r}, step={self.step_count})"
        )
