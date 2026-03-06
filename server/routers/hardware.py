"""FastAPI router for Hardware endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from server.auth import verify_auth
from server.store import HARDWARE_CATALOG

router = APIRouter(prefix="/hardware", tags=["hardware"])


# ---------------------------------------------------------------------------
# GET /hardware/catalog
# ---------------------------------------------------------------------------

@router.get("/catalog")
async def get_hardware_catalog(
    _: str = Depends(verify_auth),
) -> JSONResponse:
    return JSONResponse(HARDWARE_CATALOG)
