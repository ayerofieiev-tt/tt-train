from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta, timezone

import httpx
from sqlalchemy import select

from server.config import settings
from server.cluster import get_backend
from server.db.engine import AsyncSessionLocal
from server.db import crud
from server.db.models import Session as SessionModel

logger = logging.getLogger(__name__)


class Scheduler:
    """Async background scheduler that submits queued jobs/sessions to the
    cluster backend and handles idle-session expiry."""

    def __init__(self) -> None:
        self.backend = get_backend(settings.cluster_backend)
        self._running = False

    async def start(self) -> None:
        self._running = True
        while self._running:
            try:
                await self._tick()
            except Exception as e:
                logger.exception("Scheduler tick error: %s", e)
            await asyncio.sleep(settings.scheduler_poll_interval)

    async def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------
    # Tick — one full scheduler pass
    # ------------------------------------------------------------------

    async def _tick(self) -> None:
        async with AsyncSessionLocal() as db:
            await self._process_queued_jobs(db)
            await self._process_provisioning_sessions(db)
            await self._check_idle_sessions(db)

    # ------------------------------------------------------------------
    # Jobs
    # ------------------------------------------------------------------

    async def _process_queued_jobs(self, db) -> None:
        cap = self.backend.max_concurrent_jobs
        if cap is not None:
            running = await crud.list_jobs(db, status="running", limit=cap + 1)
            slots = cap - len(running)
            if slots <= 0:
                return
        else:
            slots = 10  # arbitrary fetch limit for Slurm (it owns scheduling)

        jobs = await crud.list_jobs(db, status="queued", limit=slots)
        for job in jobs:
            try:
                await self._submit_job(db, job)
            except Exception as e:
                logger.error("Failed to submit job %s: %s", job.id, e)
                await crud.update_job(
                    db,
                    job.id,
                    status="failed",
                    error={"type": "scheduler_error", "message": str(e)},
                )

    async def _submit_job(self, db, job) -> None:
        args = [
            "--job-id", job.id,
            "--api-url", settings.api_base_url,
            "--api-key", settings.internal_api_key,
            "--model", job.model,
            "--method", job.method,
            "--training-data", job.training_data,
            "--config", json.dumps(job.config or {}),
            "--storage-path", settings.shared_storage_path,
        ]
        if job.validation_data:
            args += ["--validation-data", job.validation_data]
        if job.console_job_id:
            args += ["--console-job-id", job.console_job_id]
        if job.dataset_url:
            args += ["--dataset-url", job.dataset_url]
        if job.console_base_url:
            args += ["--console-base-url", job.console_base_url]
        if job.worker_token:
            args += ["--worker-token", job.worker_token]
        if job.callback_url:
            args += ["--callback-url", job.callback_url]

        script = os.path.join(settings.worker_script_dir, "job_runner.py")
        nodes = (job.hardware_config or {}).get("nodes", 1)

        backend_id = await self.backend.submit_job(
            job_id=job.id,
            script_path=script,
            args=args,
            nodes=nodes,
            partition=settings.slurm_partition,
            account=settings.slurm_account,
        )

        now = datetime.now(timezone.utc)
        await crud.update_job(
            db,
            job.id,
            status="running",
            slurm_job_id=backend_id,
            started_at=now,
        )
        logger.info("Submitted job %s → backend %s", job.id, backend_id)

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------

    async def _process_provisioning_sessions(self, db) -> None:
        cap = self.backend.max_concurrent_sessions
        if cap is not None:
            running = await crud.list_sessions(db, status="ready", limit=cap + 1)
            provisioning = await crud.list_sessions(db, status="provisioning", limit=cap + 1)
            slots = cap - len(running) - len(provisioning)
            if slots <= 0:
                return
        else:
            slots = 10

        sessions = await crud.list_sessions(db, status="provisioning", limit=slots)
        for session in sessions:
            try:
                await self._submit_session(db, session)
            except Exception as e:
                logger.error("Failed to submit session %s: %s", session.id, e)
                await crud.update_session(db, session.id, status="failed")

    async def _submit_session(self, db, session) -> None:
        args = [
            "--session-id", session.id,
            "--api-url", settings.api_base_url,
            "--api-key", settings.internal_api_key,
            "--model", session.model,
            "--lora-config", json.dumps(session.lora_config or {}),
            "--optimizer-config", json.dumps(session.optimizer_config or {}),
            "--storage-path", settings.shared_storage_path,
        ]

        script = os.path.join(settings.worker_script_dir, "session_worker.py")
        nodes = (session.hardware_config or {}).get("nodes", 1)

        backend_id = await self.backend.submit_session(
            session_id=session.id,
            script_path=script,
            args=args,
            nodes=nodes,
            partition=settings.slurm_partition,
            account=settings.slurm_account,
        )

        await crud.update_session(db, session.id, slurm_job_id=backend_id)
        logger.info("Submitted session %s → backend %s", session.id, backend_id)

    # ------------------------------------------------------------------
    # Idle-session expiry
    # ------------------------------------------------------------------

    async def _check_idle_sessions(self, db) -> None:
        now = datetime.now(timezone.utc)

        result = await db.execute(
            select(SessionModel).where(
                SessionModel.status == "ready",
                SessionModel.last_active_at.is_not(None),
            )
        )
        sessions = result.scalars().all()

        for session in sessions:
            idle_minutes = session.idle_timeout_minutes or settings.session_idle_timeout_minutes
            timeout = timedelta(minutes=idle_minutes)
            # last_active_at is timezone-aware (stored with timezone=True); ensure
            # we compare against a tz-aware datetime.
            last_active = session.last_active_at
            if last_active.tzinfo is None:
                last_active = last_active.replace(tzinfo=timezone.utc)
            if (now - last_active) > timeout:
                logger.info(
                    "Session %s idle for more than %d minutes, shutting down",
                    session.id,
                    idle_minutes,
                )
                await self._shutdown_session(db, session)

    async def _shutdown_session(self, db, session: SessionModel) -> None:
        """Gracefully ask the session worker to shut down, then mark expired."""
        if session.worker_url:
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    await client.post(f"{session.worker_url}/shutdown")
            except Exception:
                # Worker may already be gone; proceed with DB update regardless.
                pass

        await crud.update_session(db, session.id, status="expired")
        logger.info("Session %s marked as expired", session.id)
