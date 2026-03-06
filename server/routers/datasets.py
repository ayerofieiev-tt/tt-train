"""FastAPI router for Datasets endpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from fastapi.responses import JSONResponse

from server.auth import error_400, error_404, verify_auth
from server.config import settings
from server.store import new_id, now_iso, store

router = APIRouter(prefix="/datasets", tags=["datasets"])


# ---------------------------------------------------------------------------
# POST /datasets
# ---------------------------------------------------------------------------

@router.post("")
async def upload_dataset(
    file: UploadFile = File(...),
    format: str = Form(...),
    name: str | None = Form(default=None),
    description: str | None = Form(default=None),
    metadata: str | None = Form(default=None),
    _: str = Depends(verify_auth),
) -> JSONResponse:
    content = await file.read()
    file_bytes = len(content)

    # Count examples by counting non-empty lines
    lines = [ln for ln in content.decode("utf-8", errors="replace").splitlines() if ln.strip()]
    examples = len(lines) if lines else 1

    tokens = examples * 100
    avg_tokens = 100
    max_tokens = 512
    min_tokens = 20

    # Parse metadata JSON string if provided
    parsed_metadata: dict[str, str] = {}
    if metadata:
        try:
            parsed_metadata = json.loads(metadata)
        except (json.JSONDecodeError, TypeError):
            raise error_400("metadata must be a valid JSON object", "invalid_metadata")

    dataset_id = new_id("ds")

    # Persist to shared storage so job_runner.py can read it
    dest = Path(settings.shared_storage_path) / "datasets" / dataset_id / "data.jsonl"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(content)

    dataset: dict[str, Any] = {
        "object": "dataset",
        "id": dataset_id,
        "name": name,
        "description": description,
        "format": format,
        "created_at": now_iso(),
        "bytes": file_bytes,
        "status": "ready",
        "stats": {
            "examples": examples,
            "tokens": tokens,
            "avg_tokens_per_example": avg_tokens,
            "max_tokens": max_tokens,
            "min_tokens": min_tokens,
        },
        "metadata": parsed_metadata,
    }

    store.datasets[dataset_id] = dataset
    return JSONResponse(dataset, status_code=201)


# ---------------------------------------------------------------------------
# GET /datasets
# ---------------------------------------------------------------------------

@router.get("")
async def list_datasets(
    limit: int = Query(default=20, ge=1, le=100),
    after: str | None = Query(default=None),
    before: str | None = Query(default=None),
    format: str | None = Query(default=None),
    _: str = Depends(verify_auth),
) -> JSONResponse:
    datasets = list(store.datasets.values())

    if format:
        datasets = [d for d in datasets if d.get("format") == format]

    ids = [d["id"] for d in datasets]

    if after and after in ids:
        idx = ids.index(after)
        datasets = datasets[idx + 1:]
        ids = ids[idx + 1:]

    if before and before in ids:
        idx = ids.index(before)
        datasets = datasets[:idx]
        ids = ids[:idx]

    has_more = len(datasets) > limit
    datasets = datasets[:limit]

    return JSONResponse({
        "object": "list",
        "data": datasets,
        "has_more": has_more,
        "first_id": datasets[0]["id"] if datasets else None,
        "last_id": datasets[-1]["id"] if datasets else None,
    })


# ---------------------------------------------------------------------------
# GET /datasets/{dataset_id}
# ---------------------------------------------------------------------------

@router.get("/{dataset_id}")
async def get_dataset(
    dataset_id: str,
    _: str = Depends(verify_auth),
) -> JSONResponse:
    dataset = store.get_dataset(dataset_id)
    if dataset is None:
        raise error_404("Dataset", dataset_id)
    return JSONResponse(dataset)


# ---------------------------------------------------------------------------
# DELETE /datasets/{dataset_id}
# ---------------------------------------------------------------------------

@router.delete("/{dataset_id}")
async def delete_dataset(
    dataset_id: str,
    _: str = Depends(verify_auth),
) -> JSONResponse:
    if dataset_id not in store.datasets:
        raise error_404("Dataset", dataset_id)
    del store.datasets[dataset_id]
    return JSONResponse({"deleted": True, "id": dataset_id})
