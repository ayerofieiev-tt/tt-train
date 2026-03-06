"""FastAPI router for Inference and Chat endpoints."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from server.auth import verify_auth
from server.store import new_id, now_iso

router = APIRouter(tags=["inference"])

# A second router for /chat prefix
chat_router = APIRouter(prefix="/chat", tags=["chat"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MOCK_RESPONSE_CONTENT = "Hello! How can I help you?"
MOCK_TOKENS = list(MOCK_RESPONSE_CONTENT.split())


def _build_inference_result(model: str, prompt_tokens: int = 20) -> dict[str, Any]:
    inf_id = new_id("inf")
    return {
        "object": "inference_result",
        "id": inf_id,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": MOCK_RESPONSE_CONTENT},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": len(MOCK_TOKENS),
        },
    }


def _estimate_prompt_tokens(messages: list[dict[str, Any]]) -> int:
    return max(
        sum(len(str(m.get("content", ""))) // 4 for m in messages),
        1,
    )


# ---------------------------------------------------------------------------
# POST /inference/generate
# ---------------------------------------------------------------------------

@router.post("/inference/generate", response_model=None)
async def inference_generate(
    body: dict[str, Any],
    _: str = Depends(verify_auth),
) -> JSONResponse | EventSourceResponse:
    model = body.get("model", "tt://catalog/llama-3.2-8b")
    messages = body.get("messages", [])
    stream = body.get("stream", False)
    prompt_tokens = _estimate_prompt_tokens(messages)

    if stream:
        async def token_generator():
            for i, token in enumerate(MOCK_TOKENS):
                chunk = {
                    "object": "inference_chunk",
                    "id": new_id("inf"),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": token + (" " if i < len(MOCK_TOKENS) - 1 else "")},
                            "finish_reason": None,
                        }
                    ],
                }
                yield {"event": "token", "data": json.dumps(chunk)}
                await asyncio.sleep(0.01)

            # Final chunk with finish_reason
            final_chunk = {
                "object": "inference_chunk",
                "id": new_id("inf"),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": len(MOCK_TOKENS),
                },
            }
            yield {"event": "token", "data": json.dumps(final_chunk)}
            yield {"event": "done", "data": "[DONE]"}

        return EventSourceResponse(token_generator())

    result = _build_inference_result(model, prompt_tokens)
    return JSONResponse(result)


# ---------------------------------------------------------------------------
# POST /inference/batch
# ---------------------------------------------------------------------------

@router.post("/inference/batch")
async def inference_batch(
    body: dict[str, Any],
    _: str = Depends(verify_auth),
) -> JSONResponse:
    model = body.get("model", "tt://catalog/llama-3.2-8b")
    requests = body.get("requests", [])

    results: list[dict[str, Any]] = []
    for req in requests:
        req_id = req.get("id", new_id("req"))
        messages = req.get("messages", [])
        prompt_tokens = _estimate_prompt_tokens(messages)
        results.append({
            "id": req_id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": MOCK_RESPONSE_CONTENT},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": len(MOCK_TOKENS),
            },
        })

    return JSONResponse({"results": results})


# ---------------------------------------------------------------------------
# POST /chat/completions  (OpenAI-compatible)
# ---------------------------------------------------------------------------

@chat_router.post("/completions")
async def chat_completions(
    body: dict[str, Any],
    _: str = Depends(verify_auth),
) -> JSONResponse:
    model = body.get("model", "tt://catalog/llama-3.2-8b")
    messages = body.get("messages", [])
    prompt_tokens = _estimate_prompt_tokens(messages)
    completion_tokens = len(MOCK_TOKENS)

    return JSONResponse({
        "id": f"chatcmpl-{new_id('mock')}",
        "object": "chat.completion",
        "created": int(__import__("time").time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": MOCK_RESPONSE_CONTENT},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    })
