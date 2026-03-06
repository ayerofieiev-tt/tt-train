"""Internal API routes — called by cluster workers, not end users.

Auth uses settings.internal_api_key instead of the user-facing API key.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, status
from fastapi.responses import JSONResponse

from server.config import settings
from server.db import crud
from server.db.engine import get_db
from server.store import new_id, now_iso
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter(prefix="/internal", tags=["internal"])


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def verify_internal_auth(authorization: str | None = Header(default=None)) -> str:
    """FastAPI dependency — validates the internal worker Bearer token."""
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": {"type": "authentication_error", "message": "Missing Authorization header"}},
        )

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": {"type": "authentication_error", "message": "Invalid Authorization header format"}},
        )

    if token.strip() != settings.internal_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": {"type": "authentication_error", "message": "Invalid internal API key"}},
        )

    return token.strip()


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# POST /internal/jobs/{job_id}/progress
# ---------------------------------------------------------------------------

@router.post("/jobs/{job_id}/progress")
async def job_progress(
    job_id: str,
    body: dict[str, Any],
    _: str = Depends(verify_internal_auth),
    db: AsyncSession = Depends(get_db),
) -> JSONResponse:
    job = await crud.get_job(db, job_id)
    if job is None:
        raise HTTPException(404, detail={"error": {"type": "not_found_error", "message": "Job not found", "id": job_id}})

    progress = {
        "step": body.get("step"),
        "total_steps": body.get("total_steps"),
        "epoch": body.get("epoch"),
        "percentage": body.get("percentage"),
        "tokens_processed": body.get("tokens_processed"),
    }
    metrics = {"train_loss": body.get("loss")}

    await crud.update_job(db, job_id, progress=progress, metrics=metrics)

    # Write structured metric row if step data is present
    step = body.get("step")
    if step is not None:
        await crud.record_metric(
            db,
            job_id=job_id,
            step=step,
            epoch=body.get("epoch"),
            train_loss=body.get("loss"),
            val_loss=body.get("val_loss"),
            grad_norm=body.get("grad_norm"),
            learning_rate=body.get("learning_rate"),
            tokens_per_second=body.get("tokens_per_second"),
        )

    return JSONResponse({"ok": True})


# ---------------------------------------------------------------------------
# POST /internal/jobs/{job_id}/logs
# ---------------------------------------------------------------------------

@router.post("/jobs/{job_id}/logs")
async def worker_append_log(
    job_id: str,
    body: dict,
    _: str = Depends(verify_internal_auth),
    db: AsyncSession = Depends(get_db),
) -> JSONResponse:
    job = await crud.get_job(db, job_id)
    if job is None:
        raise HTTPException(404, detail={"error": {"type": "not_found_error", "message": "Job not found", "id": job_id}})

    entry = await crud.append_log(
        db,
        job_id=job_id,
        log_type=body.get("log_type", "info"),
        message=body.get("message", ""),
        step=body.get("step"),
    )
    return JSONResponse({
        "id": entry.id,
        "job_id": job_id,
        "log_type": entry.log_type,
        "message": entry.message,
        "step": entry.step,
    })


# ---------------------------------------------------------------------------
# POST /internal/jobs/{job_id}/complete
# ---------------------------------------------------------------------------

@router.post("/jobs/{job_id}/complete")
async def job_complete(
    job_id: str,
    body: dict[str, Any],
    _: str = Depends(verify_internal_auth),
    db: AsyncSession = Depends(get_db),
) -> JSONResponse:
    job = await crud.get_job(db, job_id)
    if job is None:
        raise HTTPException(404, detail={"error": {"type": "not_found_error", "message": "Job not found", "id": job_id}})

    result_model = body.get("result_model")
    metrics = body.get("metrics") or {}
    completed_at = _now_utc()

    await crud.update_job(
        db,
        job_id,
        status="completed",
        result_model=result_model,
        metrics=metrics,
        completed_at=completed_at,
    )

    # Create a checkpoint record for the finished model.
    if result_model:
        ckpt_id = new_id("ckpt")
        progress = job.progress or {}
        step = progress.get("step", 0) if isinstance(progress, dict) else 0
        await crud.create_checkpoint(
            db,
            id=ckpt_id,
            model_path=result_model,
            step=step,
            job_id=job_id,
            metrics=metrics,
        )

    return JSONResponse({"ok": True})


# ---------------------------------------------------------------------------
# POST /internal/jobs/{job_id}/fail
# ---------------------------------------------------------------------------

@router.post("/jobs/{job_id}/fail")
async def job_fail(
    job_id: str,
    body: dict[str, Any],
    _: str = Depends(verify_internal_auth),
    db: AsyncSession = Depends(get_db),
) -> JSONResponse:
    job = await crud.get_job(db, job_id)
    if job is None:
        raise HTTPException(404, detail={"error": {"type": "not_found_error", "message": "Job not found", "id": job_id}})

    error = {
        "type": body.get("type", "worker_error"),
        "message": body.get("message", "Unknown error"),
        "step": body.get("step"),
    }

    await crud.update_job(
        db,
        job_id,
        status="failed",
        error=error,
        completed_at=_now_utc(),
    )
    return JSONResponse({"ok": True})


# ---------------------------------------------------------------------------
# POST /internal/sessions/{session_id}/ready
# ---------------------------------------------------------------------------

@router.post("/sessions/{session_id}/ready")
async def session_ready(
    session_id: str,
    body: dict[str, Any],
    _: str = Depends(verify_internal_auth),
    db: AsyncSession = Depends(get_db),
) -> JSONResponse:
    session = await crud.get_session(db, session_id)
    if session is None:
        raise HTTPException(404, detail={"error": {"type": "not_found_error", "message": "Session not found", "id": session_id}})

    worker_url = body.get("worker_url")
    if not worker_url:
        raise HTTPException(400, detail={"error": {"type": "invalid_request_error", "message": "worker_url is required"}})

    await crud.update_session(
        db,
        session_id,
        status="ready",
        worker_url=worker_url,
        last_active_at=_now_utc(),
    )
    return JSONResponse({"ok": True})


# ---------------------------------------------------------------------------
# POST /internal/sessions/{session_id}/heartbeat
# ---------------------------------------------------------------------------

@router.post("/sessions/{session_id}/heartbeat")
async def session_heartbeat(
    session_id: str,
    body: dict[str, Any],  # accepted but ignored
    _: str = Depends(verify_internal_auth),
    db: AsyncSession = Depends(get_db),
) -> JSONResponse:
    session = await crud.get_session(db, session_id)
    if session is None:
        raise HTTPException(404, detail={"error": {"type": "not_found_error", "message": "Session not found", "id": session_id}})

    await crud.update_session(db, session_id, last_active_at=_now_utc())
    return JSONResponse({"ok": True})
