"""Model catalog and checkpoint management."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tt_train.http import HTTPClient
from tt_train.types import ModelInfo, DownloadResult, PaginatedList


class Models:
    """Browse the model catalog, inspect checkpoints, download weights."""

    def __init__(self, http: HTTPClient):
        self._http = http

    def list(
        self,
        *,
        limit: int = 20,
        after: str | None = None,
        source: str | None = None,
        family: str | None = None,
        min_params: str | None = None,
        max_params: str | None = None,
        tags: str | None = None,
    ) -> PaginatedList:
        """List available models (catalog + user checkpoints)."""
        data = self._http.get(
            "/models",
            params={
                "limit": limit,
                "after": after,
                "source": source,
                "family": family,
                "min_params": min_params,
                "max_params": max_params,
                "tags": tags,
            },
        )
        result = PaginatedList.model_validate(data)
        result.data = [ModelInfo.model_validate(m) for m in result.data]
        return result

    def get(self, model_path: str) -> ModelInfo:
        """
        Get model details.

        Args:
            model_path: Full path like "tt://catalog/llama-3.2-8b"
                        or shorthand like "catalog/llama-3.2-8b".
        """
        encoded = self._encode_path(model_path)
        data = self._http.get(f"/models/{encoded}")
        return ModelInfo.model_validate(data)

    def download(
        self,
        model_path: str,
        *,
        path: str | Path = ".",
        format: str = "safetensors",
        component: str = "all",
    ) -> list[Path]:
        """
        Download model weights to local disk.

        Args:
            model_path: Model to download.
            path: Local directory to save files.
            format: "safetensors", "gguf", or "pytorch".
            component: "all", "lora_only", or "merged".

        Returns:
            List of downloaded file paths.
        """
        import httpx as _httpx

        encoded = self._encode_path(model_path)
        data = self._http.get(
            f"/models/{encoded}/download",
            params={"format": format, "component": component},
        )
        result = DownloadResult.model_validate(data)

        dest = Path(path)
        dest.mkdir(parents=True, exist_ok=True)
        downloaded: list[Path] = []

        for entry in result.urls:
            filepath = dest / entry.filename
            with _httpx.stream("GET", entry.url) as resp:
                resp.raise_for_status()
                with open(filepath, "wb") as f:
                    for chunk in resp.iter_bytes(chunk_size=8192):
                        f.write(chunk)
            downloaded.append(filepath)

        return downloaded

    def delete(self, model_path: str) -> dict[str, Any]:
        """Delete a user checkpoint. Catalog models cannot be deleted."""
        encoded = self._encode_path(model_path)
        return self._http.delete(f"/models/{encoded}")

    def generate(
        self,
        model: str,
        *,
        messages: list[dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs,
    ) -> dict[str, Any]:
        """Quick inference on a model. For heavier use, see Inference resource."""
        from tt_train.types import InferenceResult

        data = self._http.post(
            f"/models/{self._encode_path(model)}/generate",
            json_body={
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                **kwargs,
            },
        )
        return InferenceResult.model_validate(data)

    @staticmethod
    def _encode_path(model_path: str) -> str:
        """Encode tt:// paths for URL use."""
        import urllib.parse
        # Strip tt:// prefix for URL path
        clean = model_path.removeprefix("tt://")
        return urllib.parse.quote(clean, safe="")
