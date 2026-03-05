"""Inference API — generate from base or fine-tuned models."""

from __future__ import annotations

from typing import Any, Generator

from tt_train.http import HTTPClient
from tt_train.types import InferenceResult


class Inference:
    """Run inference on any model (base or fine-tuned)."""

    def __init__(self, http: HTTPClient):
        self._http = http

    def generate(
        self,
        model: str,
        *,
        messages: list[dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stop: list[str] | None = None,
        stream: bool = False,
        **kwargs,
    ) -> InferenceResult | Generator[dict[str, Any], None, None]:
        """
        Generate a completion.

        Args:
            model: Model path (e.g. "tt://proj_abc/ckpt_final").
            messages: Chat messages.
            max_tokens: Max tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling.
            stop: Stop sequences.
            stream: If True, returns a generator of SSE chunks.

        Returns:
            InferenceResult, or a generator if stream=True.
        """
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
            **kwargs,
        }
        if stop is not None:
            body["stop"] = stop

        if stream:
            return self._stream(body)

        data = self._http.post("/inference/generate", json_body=body)
        return InferenceResult.model_validate(data)

    def _stream(self, body: dict[str, Any]) -> Generator[dict[str, Any], None, None]:
        """Stream inference results via SSE."""
        for event in self._http.stream_sse("/inference/generate", params=None):
            yield event["data"]

    def batch(
        self,
        model: str,
        requests: list[dict[str, Any]],
        *,
        temperature: float = 0.7,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """
        Run batch inference.

        Args:
            model: Model path.
            requests: List of request dicts, each with "id" and "messages".
            temperature: Default temperature for all requests.

        Returns:
            List of result dicts with "id", "choices", and "usage".
        """
        body: dict[str, Any] = {
            "model": model,
            "requests": requests,
            "temperature": temperature,
            **kwargs,
        }
        data = self._http.post("/inference/batch", json_body=body)
        return data.get("results", [])

    def chat_completions(
        self,
        model: str,
        *,
        messages: list[dict[str, str]],
        **kwargs,
    ) -> dict[str, Any]:
        """
        OpenAI-compatible chat completions endpoint.

        Uses the same request/response format as OpenAI's API.
        """
        body = {"model": model, "messages": messages, **kwargs}
        return self._http.post("/chat/completions", json_body=body)
