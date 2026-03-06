"""FastAPI router for Sessions endpoints — backed by PostgreSQL via SQLAlchemy.

Training commands (forward_backward, step, sample, etc.) are proxied to the
session worker process running on the cluster node.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from server.auth import error_400, error_404, verify_auth
from server.db import crud
from server.db.engine import get_db
from server.store import new_id

router = APIRouter(prefix="/sessions", tags=["sessions"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dt_iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.isoformat().replace("+00:00", "Z")


def _session_to_dict(session) -> dict:
    return {
        "object": "session",
        "id": session.id,
        "model": session.model,
        "status": session.status,
        "lora": session.lora_config,
        "optimizer": session.optimizer_config,
        "hardware": session.hardware_config or {},
        "step_count": session.step_count or 0,
        "total_cost": session.total_cost or "$0.00",
        "idle_timeout_minutes": session.idle_timeout_minutes,
        "last_checkpoint": session.last_checkpoint,
        "name": session.name,
        "created_at": _dt_iso(session.created_at),
        "expires_at": _dt_iso(session.expires_at),
        "closed_at": _dt_iso(session.closed_at),
        "metadata": session.metadata_ or {},
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


def _worker_url(session) -> str:
    if not session.worker_url:
        raise HTTPException(
            503,
            detail={"error": {"type": "session_not_ready", "message": "Session worker not available"}},
        )
    return session.worker_url.rstrip("/")


async def _submit_to_worker(session, path: str, body: dict) -> dict:
    """Submit an async command. Returns {request_id} immediately."""
    url = f"{_worker_url(session)}/{path.lstrip('/')}"
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(url, json=body)
        if resp.status_code >= 400:
            raise HTTPException(resp.status_code, detail=resp.json())
        return resp.json()


async def _proxy_blocking(session, path: str, body: dict) -> dict:
    """Blocking proxy for commands where the server needs the result (save, shutdown)."""
    url = f"{_worker_url(session)}/{path.lstrip('/')}"
    async with httpx.AsyncClient(timeout=300.0) as client:
        resp = await client.post(url, json=body)
        if resp.status_code >= 400:
            raise HTTPException(resp.status_code, detail=resp.json())
        return resp.json()


async def _touch(db: AsyncSession, session_id: str) -> None:
    """Update last_active_at to now."""
    await crud.update_session(db, session_id, last_active_at=datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# POST /sessions
# ---------------------------------------------------------------------------

@router.post("", status_code=201)
async def create_session(
    body: dict[str, Any],
    _auth: str = Depends(verify_auth),
    db: AsyncSession = Depends(get_db),
) -> JSONResponse:
    model = body.get("model")
    if not model:
        raise error_400("'model' is required", "missing_model")

    session_id = new_id("sess")

    session = await crud.create_session(
        db,
        id=session_id,
        model=model,
        name=body.get("name"),
        status="provisioning",
        lora_config=body.get("lora"),
        optimizer_config=body.get("optimizer") or {"type": "adamw", "lr": 2e-5},
        hardware_config=body.get("hardware") or {"accelerator": "wormhole", "nodes": 1},
        idle_timeout_minutes=body.get("idle_timeout_minutes", 30),
        metadata_=body.get("metadata") or {},
    )

    return JSONResponse(_session_to_dict(session), status_code=201)


# ---------------------------------------------------------------------------
# GET /sessions
# ---------------------------------------------------------------------------

@router.get("")
async def list_sessions(
    limit: int = Query(default=20, ge=1, le=100),
    after: str | None = Query(default=None),
    status: str | None = Query(default=None),
    _auth: str = Depends(verify_auth),
    db: AsyncSession = Depends(get_db),
) -> JSONResponse:
    sessions = await crud.list_sessions(db, limit=limit + 1, after=after, status=status)
    has_more = len(sessions) > limit
    sessions = sessions[:limit]

    data = [_session_to_dict(s) for s in sessions]
    return JSONResponse({
        "object": "list",
        "data": data,
        "has_more": has_more,
        "first_id": data[0]["id"] if data else None,
        "last_id": data[-1]["id"] if data else None,
    })


# ---------------------------------------------------------------------------
# GET /sessions/{session_id}
# ---------------------------------------------------------------------------

@router.get("/{session_id}")
async def get_session(
    session_id: str,
    _auth: str = Depends(verify_auth),
    db: AsyncSession = Depends(get_db),
) -> JSONResponse:
    session = await crud.get_session(db, session_id)
    if session is None:
        raise error_404("session", session_id)
    return JSONResponse(_session_to_dict(session))


# ---------------------------------------------------------------------------
# DELETE /sessions/{session_id}
# ---------------------------------------------------------------------------

@router.delete("/{session_id}")
async def close_session(
    session_id: str,
    _auth: str = Depends(verify_auth),
    db: AsyncSession = Depends(get_db),
) -> JSONResponse:
    session = await crud.get_session(db, session_id)
    if session is None:
        raise error_404("session", session_id)

    # Best-effort shutdown of the worker.
    if session.worker_url:
        try:
            await _proxy_blocking(session, "/shutdown", {})
        except Exception:
            pass  # worker may already be gone

    session = await crud.update_session(
        db,
        session_id,
        status="closed",
        closed_at=datetime.now(timezone.utc),
    )
    return JSONResponse(_session_to_dict(session))


# ---------------------------------------------------------------------------
# GET /sessions/{session_id}/checkpoints
# ---------------------------------------------------------------------------

@router.get("/{session_id}/checkpoints")
async def list_checkpoints(
    session_id: str,
    _auth: str = Depends(verify_auth),
    db: AsyncSession = Depends(get_db),
) -> JSONResponse:
    session = await crud.get_session(db, session_id)
    if session is None:
        raise error_404("session", session_id)

    checkpoints = await crud.list_checkpoints_for_session(db, session_id)
    data = [_checkpoint_to_dict(c) for c in checkpoints]
    return JSONResponse({
        "object": "list",
        "data": data,
        "has_more": False,
        "first_id": data[0]["id"] if data else None,
        "last_id": data[-1]["id"] if data else None,
    })


# ---------------------------------------------------------------------------
# POST /sessions/{session_id}/forward_backward
# ---------------------------------------------------------------------------

@router.post("/{session_id}/forward_backward")
async def forward_backward(
    session_id: str,
    body: dict[str, Any],
    _auth: str = Depends(verify_auth),
    db: AsyncSession = Depends(get_db),
) -> JSONResponse:
    session = await crud.get_session(db, session_id)
    if session is None:
        raise error_404("session", session_id)
    result = await _submit_to_worker(session, "/forward_backward", body)
    await _touch(db, session_id)
    return JSONResponse(result)


# ---------------------------------------------------------------------------
# POST /sessions/{session_id}/step
# ---------------------------------------------------------------------------

@router.post("/{session_id}/step")
async def step(
    session_id: str,
    body: dict[str, Any],
    _auth: str = Depends(verify_auth),
    db: AsyncSession = Depends(get_db),
) -> JSONResponse:
    session = await crud.get_session(db, session_id)
    if session is None:
        raise error_404("session", session_id)
    result = await _submit_to_worker(session, "/step", body)
    await _touch(db, session_id)
    return JSONResponse(result)


# ---------------------------------------------------------------------------
# POST /sessions/{session_id}/sample
# ---------------------------------------------------------------------------

@router.post("/{session_id}/sample")
async def sample(
    session_id: str,
    body: dict[str, Any],
    _auth: str = Depends(verify_auth),
    db: AsyncSession = Depends(get_db),
) -> JSONResponse:
    session = await crud.get_session(db, session_id)
    if session is None:
        raise error_404("session", session_id)
    result = await _submit_to_worker(session, "/sample", body)
    await _touch(db, session_id)
    return JSONResponse(result)


# ---------------------------------------------------------------------------
# POST /sessions/{session_id}/log_probs
# ---------------------------------------------------------------------------

@router.post("/{session_id}/log_probs")
async def log_probs(
    session_id: str,
    body: dict[str, Any],
    _auth: str = Depends(verify_auth),
    db: AsyncSession = Depends(get_db),
) -> JSONResponse:
    session = await crud.get_session(db, session_id)
    if session is None:
        raise error_404("session", session_id)
    result = await _submit_to_worker(session, "/log_probs", body)
    await _touch(db, session_id)
    return JSONResponse(result)


# ---------------------------------------------------------------------------
# POST /sessions/{session_id}/eval
# ---------------------------------------------------------------------------

@router.post("/{session_id}/eval")
async def eval_session(
    session_id: str,
    body: dict[str, Any],
    _auth: str = Depends(verify_auth),
    db: AsyncSession = Depends(get_db),
) -> JSONResponse:
    session = await crud.get_session(db, session_id)
    if session is None:
        raise error_404("session", session_id)
    result = await _submit_to_worker(session, "/eval", body)
    await _touch(db, session_id)
    return JSONResponse(result)


# ---------------------------------------------------------------------------
# POST /sessions/{session_id}/retrieve
# ---------------------------------------------------------------------------

@router.post("/{session_id}/retrieve")
async def retrieve_future(
    session_id: str,
    body: dict[str, Any],
    _auth: str = Depends(verify_auth),
    db: AsyncSession = Depends(get_db),
) -> JSONResponse:
    session = await crud.get_session(db, session_id)
    if session is None:
        raise error_404("session", session_id)

    request_id = body.get("request_id")
    if not request_id:
        raise error_400("'request_id' is required", "missing_request_id")

    # Long-poll the worker: 45 s server-side wait + buffer.
    url = f"{_worker_url(session)}/retrieve"
    async with httpx.AsyncClient(timeout=50.0) as client:
        resp = await client.post(url, json={"request_id": request_id})
        if resp.status_code >= 400:
            raise HTTPException(resp.status_code, detail=resp.json())
        return JSONResponse(resp.json())


# ---------------------------------------------------------------------------
# POST /sessions/{session_id}/save
# ---------------------------------------------------------------------------

@router.post("/{session_id}/save")
async def save_checkpoint(
    session_id: str,
    body: dict[str, Any],
    _auth: str = Depends(verify_auth),
    db: AsyncSession = Depends(get_db),
) -> JSONResponse:
    session = await crud.get_session(db, session_id)
    if session is None:
        raise error_404("session", session_id)

    result = await _proxy_blocking(session, "/save", body)

    # Persist the checkpoint returned by the worker into our DB.
    ckpt_id = result.get("id") or new_id("ckpt")
    model_path = result.get("model_path", f"tt://checkpoints/{session_id}/{ckpt_id}")
    step = result.get("step", session.step_count or 0)

    ckpt = await crud.create_checkpoint(
        db,
        id=ckpt_id,
        model_path=model_path,
        step=step,
        session_id=session_id,
        name=result.get("name") or body.get("name"),
        metrics=result.get("metrics") or {},
        metadata_=result.get("metadata") or body.get("metadata") or {},
    )

    # Update session's last_checkpoint pointer.
    await crud.update_session(
        db,
        session_id,
        last_checkpoint=model_path,
        last_active_at=datetime.now(timezone.utc),
    )

    return JSONResponse(_checkpoint_to_dict(ckpt))
