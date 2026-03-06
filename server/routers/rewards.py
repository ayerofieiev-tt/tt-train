"""FastAPI router for Rewards endpoints."""

from __future__ import annotations

import random
from typing import Any

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from fastapi.responses import JSONResponse

from server.auth import error_404, verify_auth
from server.store import new_id, now_iso, store

router = APIRouter(prefix="/rewards", tags=["rewards"])


# ---------------------------------------------------------------------------
# POST /rewards
# ---------------------------------------------------------------------------

@router.post("")
async def upload_reward(
    file: UploadFile = File(...),
    runtime: str = Form(default="python3.11"),
    name: str | None = Form(default=None),
    requirements: str | None = Form(default=None),
    _: str = Depends(verify_auth),
) -> JSONResponse:
    reward_id = new_id("rf")
    reward: dict[str, Any] = {
        "object": "reward_function",
        "id": reward_id,
        "name": name,
        "status": "ready",
        "runtime": runtime,
        "created_at": now_iso(),
    }

    store.rewards[reward_id] = reward
    return JSONResponse(reward, status_code=201)


# ---------------------------------------------------------------------------
# GET /rewards
# ---------------------------------------------------------------------------

@router.get("")
async def list_rewards(
    limit: int = Query(default=20, ge=1, le=100),
    after: str | None = Query(default=None),
    _: str = Depends(verify_auth),
) -> JSONResponse:
    rewards = list(store.rewards.values())

    ids = [r["id"] for r in rewards]

    if after and after in ids:
        idx = ids.index(after)
        rewards = rewards[idx + 1:]
        ids = ids[idx + 1:]

    has_more = len(rewards) > limit
    rewards = rewards[:limit]

    return JSONResponse({
        "object": "list",
        "data": rewards,
        "has_more": has_more,
        "first_id": rewards[0]["id"] if rewards else None,
        "last_id": rewards[-1]["id"] if rewards else None,
    })


# ---------------------------------------------------------------------------
# GET /rewards/{reward_id}
# ---------------------------------------------------------------------------

@router.get("/{reward_id}")
async def get_reward(
    reward_id: str,
    _: str = Depends(verify_auth),
) -> JSONResponse:
    reward = store.get_reward(reward_id)
    if reward is None:
        raise error_404("RewardFunction", reward_id)
    return JSONResponse(reward)


# ---------------------------------------------------------------------------
# DELETE /rewards/{reward_id}
# ---------------------------------------------------------------------------

@router.delete("/{reward_id}")
async def delete_reward(
    reward_id: str,
    _: str = Depends(verify_auth),
) -> JSONResponse:
    if reward_id not in store.rewards:
        raise error_404("RewardFunction", reward_id)
    del store.rewards[reward_id]
    return JSONResponse({"deleted": True, "id": reward_id})


# ---------------------------------------------------------------------------
# POST /rewards/{reward_id}/test
# ---------------------------------------------------------------------------

@router.post("/{reward_id}/test")
async def test_reward(
    reward_id: str,
    body: dict[str, Any],
    _: str = Depends(verify_auth),
) -> JSONResponse:
    reward = store.get_reward(reward_id)
    if reward is None:
        raise error_404("RewardFunction", reward_id)

    examples = body.get("examples", [])
    results: list[dict[str, Any]] = []
    for i, example in enumerate(examples):
        score = round(random.uniform(0.0, 1.0), 4)
        results.append({
            "index": i,
            "prompt": example.get("prompt", ""),
            "completion": example.get("completion", ""),
            "score": score,
            "error": None,
        })

    return JSONResponse({
        "object": "reward_test_result",
        "results": results,
    })
