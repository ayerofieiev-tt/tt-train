"""Tests for the TT-Train SDK."""

import json
import pytest
import httpx
import respx

from tt_train.client import Client
from tt_train.errors import (
    AuthenticationError,
    InvalidRequestError,
    RateLimitError,
    NotFoundError,
)
from tt_train.types import Job, SessionInfo


BASE = "https://api.tt-train.dev/v1"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def client():
    return Client(api_key="tt-test-key", base_url=BASE)


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def test_no_api_key_raises():
    import os
    old = os.environ.pop("TT_TRAIN_API_KEY", None)
    try:
        with pytest.raises(AuthenticationError):
            Client()
    finally:
        if old:
            os.environ["TT_TRAIN_API_KEY"] = old


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

@respx.mock
def test_create_job(client):
    respx.post(f"{BASE}/jobs").mock(return_value=httpx.Response(
        201,
        json={
            "object": "job",
            "id": "job_test123",
            "name": "test-sft",
            "model": "tt://catalog/llama-3.2-8b",
            "method": "sft",
            "status": "created",
            "training_data": "ds_abc",
            "config": {"epochs": 3},
            "hardware": {"accelerator": "wormhole", "nodes": 2},
            "cost": {"estimated_total": "$12.40", "accrued": "$0.00"},
            "created_at": "2026-03-04T12:00:00Z",
            "metadata": {},
        },
    ))

    job = client.jobs.create(
        model="tt://catalog/llama-3.2-8b",
        method="sft",
        training_data="ds_abc",
        config={"epochs": 3},
    )

    assert isinstance(job, Job)
    assert job.id == "job_test123"
    assert job.status == "created"
    assert job.method == "sft"


@respx.mock
def test_get_job(client):
    respx.get(f"{BASE}/jobs/job_test123").mock(return_value=httpx.Response(
        200,
        json={
            "object": "job",
            "id": "job_test123",
            "model": "tt://catalog/llama-3.2-8b",
            "method": "sft",
            "status": "running",
            "training_data": "ds_abc",
            "progress": {
                "step": 500,
                "total_steps": 2000,
                "epoch": 0.75,
                "percentage": 25.0,
            },
            "metrics": {"train_loss": 0.42},
            "cost": {"accrued": "$3.10"},
            "created_at": "2026-03-04T12:00:00Z",
            "metadata": {},
        },
    ))

    job = client.jobs.get("job_test123")
    assert job.status == "running"
    assert job.progress.step == 500
    assert job.metrics["train_loss"] == 0.42


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

@respx.mock
def test_create_session(client):
    # Mock create
    respx.post(f"{BASE}/sessions").mock(return_value=httpx.Response(
        201,
        json={
            "object": "session",
            "id": "sess_test123",
            "model": "tt://catalog/llama-3.2-8b",
            "status": "ready",
            "lora": {"rank": 64},
            "optimizer": {"type": "adamw", "lr": 2e-5},
            "hardware": {"accelerator": "wormhole", "nodes": 1},
            "step_count": 0,
            "total_cost": "$0.00",
            "idle_timeout_minutes": 30,
            "created_at": "2026-03-04T12:00:00Z",
            "metadata": {},
        },
    ))

    # Mock the refresh call during wait_until_ready
    respx.get(f"{BASE}/sessions/sess_test123").mock(return_value=httpx.Response(
        200,
        json={
            "object": "session",
            "id": "sess_test123",
            "model": "tt://catalog/llama-3.2-8b",
            "status": "ready",
            "step_count": 0,
            "total_cost": "$0.00",
            "hardware": {},
            "created_at": "2026-03-04T12:00:00Z",
            "metadata": {},
        },
    ))

    session = client.sessions.create(
        model="tt://catalog/llama-3.2-8b",
        lora={"rank": 64},
        optimizer={"type": "adamw", "lr": 2e-5},
    )

    assert session.id == "sess_test123"
    assert session.status == "ready"


@respx.mock
def test_forward_backward(client):
    # Create session (already ready)
    respx.post(f"{BASE}/sessions").mock(return_value=httpx.Response(201, json={
        "object": "session", "id": "sess_fb", "model": "m", "status": "ready",
        "step_count": 0, "total_cost": "$0.00", "hardware": {},
        "created_at": "2026-03-04T12:00:00Z", "metadata": {},
    }))
    respx.get(f"{BASE}/sessions/sess_fb").mock(return_value=httpx.Response(200, json={
        "object": "session", "id": "sess_fb", "model": "m", "status": "ready",
        "step_count": 0, "total_cost": "$0.00", "hardware": {},
        "created_at": "2026-03-04T12:00:00Z", "metadata": {},
    }))

    respx.post(f"{BASE}/sessions/sess_fb/forward_backward").mock(
        return_value=httpx.Response(200, json={
            "object": "forward_backward_result",
            "session_id": "sess_fb",
            "loss": 2.31,
            "token_count": 1847,
            "example_count": 2,
            "grad_norm": 0.45,
            "cost": "$0.002",
            "duration_ms": 340,
        })
    )

    respx.post(f"{BASE}/sessions/sess_fb/step").mock(
        return_value=httpx.Response(200, json={
            "object": "step_result",
            "session_id": "sess_fb",
            "step_number": 1,
            "learning_rate": 2e-5,
            "grad_norm_before_clip": 0.45,
            "grad_norm_after_clip": 0.45,
            "cost": "$0.0001",
            "duration_ms": 12,
        })
    )

    session = client.sessions.create(model="m", lora={"rank": 64})

    result = session.forward_backward(
        batch=[{"messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello", "weight": 1},
        ]}],
        loss="cross_entropy",
    )
    assert result.loss == 2.31
    assert result.grad_norm == 0.45
    assert result.cost == "$0.002"

    step = session.step()
    assert step.step_number == 1
    assert session.step_count == 1


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

@respx.mock
def test_404_raises_not_found(client):
    respx.get(f"{BASE}/jobs/nonexistent").mock(return_value=httpx.Response(
        404,
        json={"error": {
            "type": "not_found_error",
            "message": "Job not found",
            "code": "job_not_found",
        }},
    ))

    with pytest.raises(NotFoundError) as exc_info:
        client.jobs.get("nonexistent")
    assert "Job not found" in str(exc_info.value)


@respx.mock
def test_rate_limit_includes_retry_after(client):
    respx.get(f"{BASE}/jobs/job_abc").mock(return_value=httpx.Response(
        429,
        json={"error": {
            "type": "rate_limit_error",
            "message": "Rate limit exceeded",
            "retry_after_seconds": 4.2,
        }},
    ))

    with pytest.raises(RateLimitError) as exc_info:
        client.jobs.get("job_abc")
    assert exc_info.value.retry_after == 4.2


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

@respx.mock
def test_upload_dataset(client, tmp_path):
    train_file = tmp_path / "train.jsonl"
    train_file.write_text(
        json.dumps({"messages": [{"role": "user", "content": "hi"}]}) + "\n"
    )

    respx.post(f"{BASE}/datasets").mock(return_value=httpx.Response(
        201,
        json={
            "object": "dataset",
            "id": "ds_test",
            "format": "chat",
            "status": "processing",
            "created_at": "2026-03-04T12:00:00Z",
        },
    ))

    ds = client.datasets.create(str(train_file), format="chat")
    assert ds.id == "ds_test"
    assert ds.status == "processing"
