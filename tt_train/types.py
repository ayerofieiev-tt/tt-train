"""Pydantic models for TT-Train API objects."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Common
# ---------------------------------------------------------------------------

class PaginatedList(BaseModel):
    object: Literal["list"] = "list"
    data: list[Any]
    has_more: bool = False
    first_id: str | None = None
    last_id: str | None = None


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class DatasetStats(BaseModel):
    examples: int
    tokens: int
    avg_tokens_per_example: int | None = None
    max_tokens: int | None = None
    min_tokens: int | None = None


class Dataset(BaseModel):
    object: Literal["dataset"] = "dataset"
    id: str
    name: str | None = None
    description: str | None = None
    format: str
    created_at: datetime
    bytes: int | None = None
    status: str
    stats: DatasetStats | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class HardwareRecommendation(BaseModel):
    accelerator: str
    min_nodes: int


class ModelInfo(BaseModel):
    object: Literal["model"] = "model"
    id: str
    name: str
    family: str | None = None
    params: str | None = None
    active_params: str | None = None
    architecture: str | None = None
    max_seq_len: int | None = None
    source: str  # "catalog" | "checkpoint"
    supported_methods: list[str] = Field(default_factory=list)
    supported_hardware: list[str] = Field(default_factory=list)
    recommended_hardware: dict[str, HardwareRecommendation] = Field(default_factory=dict)
    license: str | None = None
    created_at: datetime | None = None

    # Checkpoint-specific
    base_model: str | None = None
    method: str | None = None
    job_id: str | None = None
    session_id: str | None = None
    step: int | None = None
    lora: dict[str, Any] | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    parent_checkpoint: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class DownloadURL(BaseModel):
    filename: str
    url: str
    bytes: int
    expires_at: datetime


class DownloadResult(BaseModel):
    object: Literal["download"] = "download"
    urls: list[DownloadURL]


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

class JobProgress(BaseModel):
    step: int
    total_steps: int
    epoch: float
    percentage: float
    tokens_processed: int | None = None
    examples_processed: int | None = None


class JobCost(BaseModel):
    estimated_total: str | None = None
    estimated_time_seconds: int | None = None
    accrued: str | None = None
    elapsed_seconds: int | None = None


class JobError(BaseModel):
    type: str
    message: str
    code: str | None = None
    step: int | None = None
    last_checkpoint: str | None = None


class Job(BaseModel):
    object: Literal["job"] = "job"
    id: str
    name: str | None = None
    model: str
    method: str
    status: str
    training_data: str
    validation_data: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    hardware: dict[str, Any] = Field(default_factory=dict)
    progress: JobProgress | None = None
    metrics: dict[str, Any] | None = None
    result_model: str | None = None
    error: JobError | None = None
    cost: JobCost | None = None
    created_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class Checkpoint(BaseModel):
    object: Literal["checkpoint"] = "checkpoint"
    id: str
    model_path: str
    job_id: str | None = None
    session_id: str | None = None
    step: int
    epoch: float | None = None
    name: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None


class JobEstimate(BaseModel):
    object: Literal["estimate"] = "estimate"
    estimated_cost: str
    estimated_time_seconds: int
    estimated_steps: int
    tokens_total: int
    hardware_plan: dict[str, Any] = Field(default_factory=dict)
    cost_breakdown: dict[str, Any] = Field(default_factory=dict)
    queue_estimate: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

class SessionInfo(BaseModel):
    object: Literal["session"] = "session"
    id: str
    model: str
    status: str
    lora: dict[str, Any] | None = None
    optimizer: dict[str, Any] | None = None
    hardware: dict[str, Any] = Field(default_factory=dict)
    step_count: int = 0
    total_cost: str = "$0.00"
    idle_timeout_minutes: int | None = None
    last_checkpoint: str | None = None
    name: str | None = None
    created_at: datetime | None = None
    expires_at: datetime | None = None
    closed_at: datetime | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class SampleOutput(BaseModel):
    index: int
    text: str
    tokens: int
    finish_reason: str
    log_prob: float | None = None
    per_token_log_probs: list[dict[str, Any]] = Field(default_factory=list)


class SampleCompletion(BaseModel):
    prompt_index: int
    outputs: list[SampleOutput]


class SampleResult(BaseModel):
    object: Literal["sample_result"] = "sample_result"
    session_id: str
    completions: list[SampleCompletion]
    usage: dict[str, int] = Field(default_factory=dict)
    cost: str | None = None


class ForwardBackwardResult(BaseModel):
    object: Literal["forward_backward_result"] = "forward_backward_result"
    session_id: str
    loss: float
    token_count: int
    example_count: int
    grad_norm: float | None = None
    cost: str | None = None
    duration_ms: int | None = None


class StepResult(BaseModel):
    object: Literal["step_result"] = "step_result"
    session_id: str
    step_number: int
    learning_rate: float | None = None
    grad_norm_before_clip: float | None = None
    grad_norm_after_clip: float | None = None
    cost: str | None = None
    duration_ms: int | None = None


class LogProbScore(BaseModel):
    index: int
    total_log_prob: float
    avg_log_prob: float
    tokens: int
    per_token: list[dict[str, Any]] = Field(default_factory=list)


class LogProbsResult(BaseModel):
    object: Literal["log_probs_result"] = "log_probs_result"
    session_id: str
    scores: list[LogProbScore]
    cost: str | None = None


class EvalResult(BaseModel):
    object: Literal["eval_result"] = "eval_result"
    session_id: str
    step: int | None = None
    examples_evaluated: int
    metrics: dict[str, float] = Field(default_factory=dict)
    cost: str | None = None
    duration_ms: int | None = None


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

class InferenceChoice(BaseModel):
    index: int = 0
    message: dict[str, str]
    finish_reason: str


class InferenceResult(BaseModel):
    object: Literal["inference_result"] = "inference_result"
    id: str
    model: str
    choices: list[InferenceChoice]
    usage: dict[str, int] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------

class RewardFunction(BaseModel):
    object: Literal["reward_function"] = "reward_function"
    id: str
    name: str | None = None
    status: str
    runtime: str | None = None
    created_at: datetime | None = None


class RewardTestResult(BaseModel):
    object: Literal["reward_test_result"] = "reward_test_result"
    results: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------

class AcceleratorPricing(BaseModel):
    per_node_hour: str
    interruption_probability: str | None = None


class Accelerator(BaseModel):
    id: str
    name: str
    memory_per_device_gb: int
    devices_per_node: int
    interconnect: str | None = None
    best_for: str | None = None
    available_nodes: int
    pricing: dict[str, AcceleratorPricing] = Field(default_factory=dict)


class HardwareCatalog(BaseModel):
    object: Literal["hardware_catalog"] = "hardware_catalog"
    accelerators: list[Accelerator]
    queue: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Events (SSE)
# ---------------------------------------------------------------------------

class JobEvent(BaseModel):
    event: str
    data: dict[str, Any] = Field(default_factory=dict)
