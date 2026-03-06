"""TT-Train API server.

Run with:
    uvicorn server.main:app --reload --port 8000

Environment variables (see server/config.py for full list):
    TT_TRAIN_DATABASE_URL=postgresql+asyncpg://tt_train:tt_train@localhost:5432/tt_train
    TT_TRAIN_CLUSTER_BACKEND=local   # or "slurm"
    TT_TRAIN_API_BASE_URL=http://localhost:8000
    TT_TRAIN_INTERNAL_API_KEY=internal-secret
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

app = FastAPI(
    title="TT-Train API Server",
    description="Fine-tuning and training on Tenstorrent hardware.",
    version="0.1.0",
)


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup():
    # Create DB tables (replace with Alembic in production)
    try:
        from server.db.engine import create_tables
        await create_tables()
        logger.info("Database tables ready.")
    except Exception as e:
        logger.warning("DB init failed (is Postgres running?): %s", e)

    # Start scheduler in background
    try:
        from server.scheduler.service import Scheduler
        app.state.scheduler = Scheduler()
        app.state.scheduler_task = asyncio.create_task(app.state.scheduler.start())
        logger.info("Scheduler started.")
    except Exception as e:
        logger.warning("Scheduler failed to start: %s", e)


@app.on_event("shutdown")
async def shutdown():
    if hasattr(app.state, "scheduler"):
        await app.state.scheduler.stop()
    if hasattr(app.state, "scheduler_task"):
        app.state.scheduler_task.cancel()


# ---------------------------------------------------------------------------
# Mount routers
# ---------------------------------------------------------------------------

from server.routers import datasets, hardware, inference, internal, jobs, models, rewards, sessions  # noqa: E402

app.include_router(jobs.router, prefix="/v1")
app.include_router(sessions.router, prefix="/v1")
app.include_router(datasets.router, prefix="/v1")
app.include_router(models.router, prefix="/v1")
app.include_router(inference.router, prefix="/v1")
app.include_router(inference.chat_router, prefix="/v1")
app.include_router(rewards.router, prefix="/v1")
app.include_router(hardware.router, prefix="/v1")
app.include_router(internal.router, prefix="/v1")  # internal worker callbacks


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": {"type": "not_found_error", "message": "Not found"}},
    )


@app.exception_handler(405)
async def method_not_allowed_handler(request: Request, exc):
    return JSONResponse(
        status_code=405,
        content={"error": {"type": "invalid_request_error", "message": "Method not allowed"}},
    )


@app.get("/v1/health", include_in_schema=False)
async def health():
    return {"status": "ok"}
