"""FastAPI router for Jobs endpoints — backed by PostgreSQL via SQLAlchemy."""

from __future__ import annotations

import asyncio
import json
import random
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from server.auth import error_400, error_404, verify_auth
from server.db import crud
from server.db.engine import get_db
from server.store import new_id, now_iso

router = APIRouter(prefix="/jobs", tags=["jobs"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dt_iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.isoformat().replace("+00:00", "Z")


def _job_to_dict(job) -> dict:
    return {
        "object": "job",
        "id": job.id,
        "name": job.name,
        "model": job.model,
        "method": job.method,
        "status": job.status,
        "training_data": job.training_data,
        "validation_data": job.validation_data,
        "config": job.config or {},
        "hardware": job.hardware_config or {},
        "progress": job.progress,
        "metrics": job.metrics,
        "result_model": job.result_model,
        "error": job.error,
        "cost": job.cost,
        "created_at": _dt_iso(job.created_at),
        "started_at": _dt_iso(job.started_at),
        "completed_at": _dt_iso(job.completed_at),
        "metadata": job.metadata_ or {},
        "console_job_id": job.console_job_id,
        "dataset_url": job.dataset_url,
    }


def _checkpoint_to_dict(ckpt) -> dict:
    return {
        "object": "checkpoint",
        "id": ckpt.id,
        "model_path": ckpt.model_path,
        "job_id": ckpt.job_id,
        "session_id": ckpt.session_id,
        "step": ckpt.step,
        "epoch": ckpt.epoch,
        "name": ckpt.name,
        "metrics": ckpt.metrics or {},
        "metadata": ckpt.metadata_ or {},
        "created_at": _dt_iso(ckpt.created_at),
    }


def _cost_estimate(method: str) -> dict[str, Any]:
    if method == "rl":
        estimated_total = f"${random.uniform(50, 200):.2f}"
        estimated_time = random.randint(7200, 28800)
    else:
        estimated_total = f"${random.uniform(10, 50):.2f}"
        estimated_time = random.randint(1800, 7200)
    return {
        "estimated_total": estimated_total,
        "estimated_time_seconds": estimated_time,
        "accrued": "$0.00",
        "elapsed_seconds": 0,
    }


# ---------------------------------------------------------------------------
# POST /jobs/estimate  (must be before /{job_id} routes)
# ---------------------------------------------------------------------------

@router.post("/estimate")
async def estimate_job(
    body: dict[str, Any],
    _: str = Depends(verify_auth),
) -> JSONResponse:
    method = body.get("method", "sft")
    config = body.get("config") or {}
    hardware = body.get("hardware") or {}

    epochs = config.get("epochs", 3)
    tokens_total = 1_000_000 * epochs

    if method == "rl":
        est_cost = random.uniform(50, 200)
        est_time = random.randint(7200, 28800)
        est_steps = random.randint(5000, 20000)
        nodes = hardware.get("min_nodes", 2)
        per_node = 2.40
    else:
        est_cost = random.uniform(10, 50)
        est_time = random.randint(1800, 7200)
        est_steps = random.randint(1000, 5000)
        nodes = hardware.get("min_nodes", 1)
        per_node = 2.40

    hours = est_time / 3600
    compute_cost = nodes * per_node * hours
    storage_cost = 0.50

    result = {
        "object": "estimate",
        "estimated_cost": f"${est_cost:.2f}",
        "estimated_time_seconds": est_time,
        "estimated_steps": est_steps,
        "tokens_total": tokens_total,
        "hardware_plan": {
            "accelerator": hardware.get("accelerator", "wormhole"),
            "nodes": nodes,
            "pricing_tier": hardware.get("pricing_tier", "on_demand"),
        },
        "cost_breakdown": {
            "compute": f"${compute_cost:.2f}",
            "storage": f"${storage_cost:.2f}",
            "data_transfer": "$0.00",
        },
        "queue_estimate": {
            "estimated_wait_minutes": 2,
            "priority": "normal",
        },
    }
    return JSONResponse(result, status_code=200)


# ---------------------------------------------------------------------------
# POST /jobs
# ---------------------------------------------------------------------------

@router.post("")
async def create_job(
    body: dict[str, Any],
    _: str = Depends(verify_auth),
    db: AsyncSession = Depends(get_db),
) -> JSONResponse:
    model = body.get("model")
    method = body.get("method")
    training_data = body.get("training_data")
    dataset_url = body.get("dataset_url")

    if not model or not method or (not training_data and not dataset_url):
        raise error_400("model, method, and either training_data or dataset_url are required")

    job_id = new_id("job")

    # Derive a fallback training_data identifier from dataset_url when not provided directly.
    if not training_data and dataset_url:
        last_segment = dataset_url.rstrip("/").rsplit("/", 1)[-1]
        training_data = last_segment or job_id

    job = await crud.create_job(
        db,
        id=job_id,
        model=model,
        method=method,
        training_data=training_data or job_id,
        name=body.get("name"),
        validation_data=body.get("validation_data"),
        config=body.get("config") or {},
        hardware_config=body.get("hardware") or {},
        metadata_=body.get("metadata") or {},
        status="queued",
        cost=_cost_estimate(method),
        # console integration fields (all optional)
        console_job_id=body.get("console_job_id"),
        dataset_url=dataset_url,
        console_base_url=body.get("console_base_url"),
        worker_token=body.get("worker_token"),
        callback_url=body.get("callback_url"),
    )

    return JSONResponse(_job_to_dict(job), status_code=201)


# ---------------------------------------------------------------------------
# GET /jobs
# ---------------------------------------------------------------------------

@router.get("")
async def list_jobs(
    limit: int = Query(default=20, ge=1, le=100),
    after: str | None = Query(default=None),
    status: str | None = Query(default=None),
    method: str | None = Query(default=None),
    _: str = Depends(verify_auth),
    db: AsyncSession = Depends(get_db),
) -> JSONResponse:
    # Fetch one extra to determine has_more.
    jobs = await crud.list_jobs(db, limit=limit + 1, after=after, status=status, method=method)
    has_more = len(jobs) > limit
    jobs = jobs[:limit]

    data = [_job_to_dict(j) for j in jobs]
    return JSONResponse({
        "object": "list",
        "data": data,
        "has_more": has_more,
        "first_id": data[0]["id"] if data else None,
        "last_id": data[-1]["id"] if data else None,
    })


# ---------------------------------------------------------------------------
# GET /jobs/{job_id}
# ---------------------------------------------------------------------------

@router.get("/{job_id}")
async def get_job(
    job_id: str,
    _: str = Depends(verify_auth),
    db: AsyncSession = Depends(get_db),
) -> JSONResponse:
    job = await crud.get_job(db, job_id)
    if job is None:
        raise error_404("Job", job_id)
    return JSONResponse(_job_to_dict(job))


# ---------------------------------------------------------------------------
# POST /jobs/{job_id}/cancel
# ---------------------------------------------------------------------------

@router.post("/{job_id}/cancel")
async def cancel_job(
    job_id: str,
    _: str = Depends(verify_auth),
    db: AsyncSession = Depends(get_db),
) -> JSONResponse:
    job = await crud.get_job(db, job_id)
    if job is None:
        raise error_404("Job", job_id)

    job = await crud.update_job(
        db,
        job_id,
        status="cancelled",
        completed_at=datetime.now(timezone.utc),
    )
    return JSONResponse(_job_to_dict(job))


# ---------------------------------------------------------------------------
# POST /jobs/{job_id}/pause
# ---------------------------------------------------------------------------

@router.post("/{job_id}/pause")
async def pause_job(
    job_id: str,
    _: str = Depends(verify_auth),
    db: AsyncSession = Depends(get_db),
) -> JSONResponse:
    job = await crud.get_job(db, job_id)
    if job is None:
        raise error_404("Job", job_id)

    job = await crud.update_job(db, job_id, status="paused")
    return JSONResponse(_job_to_dict(job))


# ---------------------------------------------------------------------------
# POST /jobs/{job_id}/resume
# ---------------------------------------------------------------------------

@router.post("/{job_id}/resume")
async def resume_job(
    job_id: str,
    body: dict[str, Any] | None = None,
    _: str = Depends(verify_auth),
    db: AsyncSession = Depends(get_db),
) -> JSONResponse:
    job = await crud.get_job(db, job_id)
    if job is None:
        raise error_404("Job", job_id)

    updates: dict[str, Any] = {"status": "queued"}
    if body and body.get("hardware"):
        updates["hardware_config"] = body["hardware"]

    job = await crud.update_job(db, job_id, **updates)
    return JSONResponse(_job_to_dict(job))


# ---------------------------------------------------------------------------
# GET /jobs/{job_id}/checkpoints
# ---------------------------------------------------------------------------

@router.get("/{job_id}/checkpoints")
async def list_checkpoints(
    job_id: str,
    _: str = Depends(verify_auth),
    db: AsyncSession = Depends(get_db),
) -> JSONResponse:
    job = await crud.get_job(db, job_id)
    if job is None:
        raise error_404("Job", job_id)

    checkpoints = await crud.list_checkpoints_for_job(db, job_id)
    data = [_checkpoint_to_dict(c) for c in checkpoints]
    return JSONResponse({
        "object": "list",
        "data": data,
        "has_more": False,
        "first_id": data[0]["id"] if data else None,
        "last_id": data[-1]["id"] if data else None,
    })


# ---------------------------------------------------------------------------
# GET /jobs/{job_id}/metrics
# ---------------------------------------------------------------------------

@router.get("/{job_id}/metrics")
async def list_metrics(
    job_id: str,
    _: str = Depends(verify_auth),
    db: AsyncSession = Depends(get_db),
) -> JSONResponse:
    job = await crud.get_job(db, job_id)
    if job is None:
        raise error_404("Job", job_id)

    metrics = await crud.list_metrics_for_job(db, job_id)
    data = [
        {
            "step": m.step,
            "epoch": m.epoch,
            "train_loss": m.train_loss,
            "val_loss": m.val_loss,
            "grad_norm": m.grad_norm,
            "learning_rate": m.learning_rate,
            "tokens_per_second": m.tokens_per_second,
            "recorded_at": m.recorded_at.isoformat().replace("+00:00", "Z") if m.recorded_at else None,
        }
        for m in metrics
    ]
    return JSONResponse({
        "object": "list",
        "data": data,
        "has_more": False,
        "job_id": job_id,
    })


# ---------------------------------------------------------------------------
# GET /jobs/{job_id}/logs
# ---------------------------------------------------------------------------

@router.get("/{job_id}/logs")
async def list_logs(
    job_id: str,
    log_type: str | None = Query(default=None),
    limit: int = Query(default=200, ge=1, le=1000),
    _: str = Depends(verify_auth),
    db: AsyncSession = Depends(get_db),
) -> JSONResponse:
    job = await crud.get_job(db, job_id)
    if job is None:
        raise error_404("Job", job_id)

    logs = await crud.list_logs_for_job(db, job_id, log_type=log_type, limit=limit)
    data = [
        {
            "id": e.id,
            "step": e.step,
            "log_type": e.log_type,
            "message": e.message,
            "logged_at": e.logged_at.isoformat().replace("+00:00", "Z") if e.logged_at else None,
        }
        for e in logs
    ]
    return JSONResponse({
        "object": "list",
        "data": data,
        "has_more": len(data) == limit,
        "job_id": job_id,
    })


# ---------------------------------------------------------------------------
# GET /jobs/{job_id}/events  — SSE stream
# ---------------------------------------------------------------------------

@router.get("/{job_id}/events")
async def stream_job_events(
    job_id: str,
    after_step: int | None = Query(default=None),
    _: str = Depends(verify_auth),
    db: AsyncSession = Depends(get_db),
) -> EventSourceResponse:
    job = await crud.get_job(db, job_id)
    if job is None:
        raise error_404("Job", job_id)

    async def generator():
        # For terminal states, emit a single state event and close.
        terminal = job.status in ("completed", "failed", "cancelled")

        if terminal:
            event_name = f"job.{job.status}"
            data: dict[str, Any] = {"job_id": job_id, "status": job.status}
            if job.status == "completed":
                data["completed_at"] = _dt_iso(job.completed_at)
                data["result_model"] = job.result_model
            elif job.status == "failed":
                data["error"] = job.error
            if job.progress and (after_step is None or (job.progress.get("step", 0) > after_step)):
                yield {
                    "event": "job.progress",
                    "data": json.dumps(job.progress),
                }
                await asyncio.sleep(0)
            yield {"event": event_name, "data": json.dumps(data)}
            await asyncio.sleep(0)
            yield {"event": "done", "data": "[DONE]"}
            return

        # For non-terminal states, poll DB every 2 seconds and emit updates.
        last_step_seen = after_step
        poll_db = crud  # capture reference

        for _ in range(150):  # up to ~5 minutes of polling
            # Re-fetch from a fresh session each iteration via a new query.
            current = await poll_db.get_job(db, job_id)
            if current is None:
                break

            if current.progress:
                step = current.progress.get("step", 0) if isinstance(current.progress, dict) else 0
                if last_step_seen is None or step > last_step_seen:
                    yield {
                        "event": "job.progress",
                        "data": json.dumps(current.progress),
                    }
                    last_step_seen = step

            if current.status in ("completed", "failed", "cancelled"):
                event_name = f"job.{current.status}"
                data = {"job_id": job_id, "status": current.status}
                if current.status == "completed":
                    data["completed_at"] = _dt_iso(current.completed_at)
                    data["result_model"] = current.result_model
                elif current.status == "failed":
                    data["error"] = current.error
                yield {"event": event_name, "data": json.dumps(data)}
                break

            await asyncio.sleep(2)

        yield {"event": "done", "data": "[DONE]"}

    return EventSourceResponse(generator())
