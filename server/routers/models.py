"""FastAPI router for Models endpoints."""

from __future__ import annotations

import urllib.parse
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse

from server.auth import error_400, error_404, verify_auth
from server.store import CATALOG_MODELS, new_id, now_iso, store

router = APIRouter(prefix="/models", tags=["models"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decode_model_path(encoded: str) -> str:
    """Decode URL-encoded model path and re-add tt:// prefix."""
    decoded = urllib.parse.unquote(encoded)
    if not decoded.startswith("tt://"):
        decoded = "tt://" + decoded
    return decoded


def _find_model(model_id: str) -> dict | None:
    """Look up a model by its tt:// ID in catalog or store checkpoints."""
    for m in CATALOG_MODELS:
        if m["id"] == model_id:
            return dict(m)
    # Check user checkpoints stored from job/session completions
    for checkpoints in store.job_checkpoints.values():
        for ckpt in checkpoints:
            model_path = ckpt.get("model_path")
            if model_path == model_id:
                return _checkpoint_to_model(ckpt)
    for checkpoints in store.session_checkpoints.values():
        for ckpt in checkpoints:
            model_path = ckpt.get("model_path")
            if model_path == model_id:
                return _checkpoint_to_model(ckpt)
    return None


def _checkpoint_to_model(ckpt: dict) -> dict:
    """Convert a checkpoint dict to a ModelInfo-shaped dict."""
    return {
        "object": "model",
        "id": ckpt.get("model_path", ckpt.get("id", "")),
        "name": ckpt.get("name") or ckpt.get("id", ""),
        "family": None,
        "params": None,
        "active_params": None,
        "architecture": None,
        "max_seq_len": None,
        "source": "checkpoint",
        "supported_methods": [],
        "supported_hardware": [],
        "recommended_hardware": {},
        "license": None,
        "created_at": ckpt.get("created_at"),
        "base_model": None,
        "method": None,
        "job_id": ckpt.get("job_id"),
        "session_id": ckpt.get("session_id"),
        "step": ckpt.get("step"),
        "lora": None,
        "metrics": ckpt.get("metrics", {}),
        "parent_checkpoint": None,
        "metadata": ckpt.get("metadata", {}),
    }


def _all_models() -> list[dict]:
    """Return all models: catalog + user checkpoints."""
    models: list[dict] = [dict(m) for m in CATALOG_MODELS]
    for checkpoints in store.job_checkpoints.values():
        for ckpt in checkpoints:
            models.append(_checkpoint_to_model(ckpt))
    for checkpoints in store.session_checkpoints.values():
        for ckpt in checkpoints:
            models.append(_checkpoint_to_model(ckpt))
    return models


# ---------------------------------------------------------------------------
# GET /models
# ---------------------------------------------------------------------------

@router.get("")
async def list_models(
    limit: int = Query(default=20, ge=1, le=100),
    after: str | None = Query(default=None),
    source: str | None = Query(default=None),
    family: str | None = Query(default=None),
    min_params: str | None = Query(default=None),
    max_params: str | None = Query(default=None),
    tags: str | None = Query(default=None),
    _: str = Depends(verify_auth),
) -> JSONResponse:
    models = _all_models()

    if source:
        models = [m for m in models if m.get("source") == source]
    if family:
        models = [m for m in models if m.get("family") == family]

    ids = [m["id"] for m in models]

    if after and after in ids:
        idx = ids.index(after)
        models = models[idx + 1:]
        ids = ids[idx + 1:]

    has_more = len(models) > limit
    models = models[:limit]

    return JSONResponse({
        "object": "list",
        "data": models,
        "has_more": has_more,
        "first_id": models[0]["id"] if models else None,
        "last_id": models[-1]["id"] if models else None,
    })


# ---------------------------------------------------------------------------
# GET /models/{model_path:path}
# ---------------------------------------------------------------------------

@router.get("/{model_path:path}/download")
async def get_model_download(
    model_path: str,
    format: str = Query(default="safetensors"),
    component: str = Query(default="all"),
    _: str = Depends(verify_auth),
) -> JSONResponse:
    model_id = _decode_model_path(model_path)
    model = _find_model(model_id)
    if model is None:
        raise error_404("Model", model_id)

    expires_at = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    safe_name = model_id.replace("tt://", "").replace("/", "_")

    urls: list[dict[str, Any]] = [
        {
            "filename": f"{safe_name}.{format}",
            "url": f"https://mock-storage.tenstorrent.com/models/{safe_name}/model.{format}?token=mock_presigned_token_1",
            "bytes": 8_000_000_000,
            "expires_at": expires_at,
        },
        {
            "filename": "config.json",
            "url": f"https://mock-storage.tenstorrent.com/models/{safe_name}/config.json?token=mock_presigned_token_2",
            "bytes": 4096,
            "expires_at": expires_at,
        },
        {
            "filename": "tokenizer.model",
            "url": f"https://mock-storage.tenstorrent.com/models/{safe_name}/tokenizer.model?token=mock_presigned_token_3",
            "bytes": 499_723,
            "expires_at": expires_at,
        },
    ]

    return JSONResponse({
        "object": "download",
        "urls": urls,
    })


@router.get("/{model_path:path}")
async def get_model(
    model_path: str,
    _: str = Depends(verify_auth),
) -> JSONResponse:
    model_id = _decode_model_path(model_path)
    model = _find_model(model_id)
    if model is None:
        raise error_404("Model", model_id)
    return JSONResponse(model)


# ---------------------------------------------------------------------------
# DELETE /models/{model_path:path}
# ---------------------------------------------------------------------------

@router.delete("/{model_path:path}")
async def delete_model(
    model_path: str,
    _: str = Depends(verify_auth),
) -> JSONResponse:
    model_id = _decode_model_path(model_path)
    model = _find_model(model_id)
    if model is None:
        raise error_404("Model", model_id)

    if model.get("source") != "checkpoint":
        raise error_400("Catalog models cannot be deleted", "cannot_delete_catalog_model")

    # Remove from job_checkpoints
    for job_id, checkpoints in store.job_checkpoints.items():
        store.job_checkpoints[job_id] = [
            c for c in checkpoints if c.get("model_path") != model_id
        ]
    # Remove from session_checkpoints
    for session_id, checkpoints in store.session_checkpoints.items():
        store.session_checkpoints[session_id] = [
            c for c in checkpoints if c.get("model_path") != model_id
        ]

    return JSONResponse({"deleted": True, "id": model_id})


# ---------------------------------------------------------------------------
# POST /models/{model_path:path}/generate
# ---------------------------------------------------------------------------

@router.post("/{model_path:path}/generate")
async def model_generate(
    model_path: str,
    body: dict[str, Any],
    _: str = Depends(verify_auth),
) -> JSONResponse:
    model_id = _decode_model_path(model_path)
    model = _find_model(model_id)
    if model is None:
        raise error_404("Model", model_id)

    messages = body.get("messages", [])
    prompt_tokens = sum(len(str(m.get("content", ""))) // 4 for m in messages) or 20

    inf_id = new_id("inf")
    result: dict[str, Any] = {
        "object": "inference_result",
        "id": inf_id,
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello! How can I help you?"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": 10,
        },
    }
    return JSONResponse(result)
