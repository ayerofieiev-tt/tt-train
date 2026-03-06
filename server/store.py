"""In-memory state store for the TT-Train mock server."""

from __future__ import annotations

import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class Store:
    jobs: dict[str, dict] = field(default_factory=dict)
    sessions: dict[str, dict] = field(default_factory=dict)
    datasets: dict[str, dict] = field(default_factory=dict)
    rewards: dict[str, dict] = field(default_factory=dict)
    # keyed by job_id -> list of checkpoint dicts
    job_checkpoints: dict[str, list] = field(default_factory=dict)
    # keyed by session_id -> list of checkpoint dicts
    session_checkpoints: dict[str, list] = field(default_factory=dict)
    # keyed by job_id -> list of SSE event dicts (for replay / streaming)
    job_events: dict[str, list] = field(default_factory=dict)

    def get_job(self, job_id: str) -> dict | None:
        return deepcopy(self.jobs.get(job_id))

    def get_session(self, session_id: str) -> dict | None:
        return deepcopy(self.sessions.get(session_id))

    def get_dataset(self, dataset_id: str) -> dict | None:
        return deepcopy(self.datasets.get(dataset_id))

    def get_reward(self, reward_id: str) -> dict | None:
        return deepcopy(self.rewards.get(reward_id))


# Singleton store instance
store = Store()

# ---------------------------------------------------------------------------
# Pre-populate model catalog
# ---------------------------------------------------------------------------

CATALOG_MODELS: list[dict] = [
    {
        "object": "model",
        "id": "tt://catalog/llama-3.2-8b",
        "name": "Llama 3.2 8B",
        "family": "llama",
        "params": "8B",
        "active_params": "8B",
        "architecture": "decoder",
        "max_seq_len": 131072,
        "source": "catalog",
        "supported_methods": ["sft", "dpo", "rl", "pretrain"],
        "supported_hardware": ["wormhole", "blackhole"],
        "recommended_hardware": {
            "sft": {"accelerator": "wormhole", "min_nodes": 1},
            "rl": {"accelerator": "wormhole", "min_nodes": 2},
        },
        "license": "llama3",
        "created_at": "2024-09-25T00:00:00Z",
        "base_model": None,
        "method": None,
        "job_id": None,
        "session_id": None,
        "step": None,
        "lora": None,
        "metrics": {},
        "parent_checkpoint": None,
        "metadata": {},
    },
    {
        "object": "model",
        "id": "tt://catalog/llama-3.1-70b",
        "name": "Llama 3.1 70B",
        "family": "llama",
        "params": "70B",
        "active_params": "70B",
        "architecture": "decoder",
        "max_seq_len": 131072,
        "source": "catalog",
        "supported_methods": ["sft", "dpo", "rl", "pretrain"],
        "supported_hardware": ["wormhole", "blackhole"],
        "recommended_hardware": {
            "sft": {"accelerator": "wormhole", "min_nodes": 4},
            "rl": {"accelerator": "wormhole", "min_nodes": 8},
        },
        "license": "llama3",
        "created_at": "2024-07-23T00:00:00Z",
        "base_model": None,
        "method": None,
        "job_id": None,
        "session_id": None,
        "step": None,
        "lora": None,
        "metrics": {},
        "parent_checkpoint": None,
        "metadata": {},
    },
    {
        "object": "model",
        "id": "tt://catalog/mistral-7b-v0.3",
        "name": "Mistral 7B v0.3",
        "family": "mistral",
        "params": "7B",
        "active_params": "7B",
        "architecture": "decoder",
        "max_seq_len": 32768,
        "source": "catalog",
        "supported_methods": ["sft", "dpo", "pretrain"],
        "supported_hardware": ["wormhole"],
        "recommended_hardware": {
            "sft": {"accelerator": "wormhole", "min_nodes": 1},
        },
        "license": "apache-2.0",
        "created_at": "2024-05-22T00:00:00Z",
        "base_model": None,
        "method": None,
        "job_id": None,
        "session_id": None,
        "step": None,
        "lora": None,
        "metrics": {},
        "parent_checkpoint": None,
        "metadata": {},
    },
]

HARDWARE_CATALOG: dict = {
    "object": "hardware_catalog",
    "accelerators": [
        {
            "id": "wormhole",
            "name": "Tenstorrent Wormhole n300",
            "memory_per_device_gb": 12,
            "devices_per_node": 8,
            "interconnect": "ethernet",
            "best_for": "Training and fine-tuning LLMs up to 70B parameters",
            "available_nodes": 64,
            "pricing": {
                "on_demand": {"per_node_hour": "$2.40", "interruption_probability": None},
                "spot": {"per_node_hour": "$0.96", "interruption_probability": "~15%"},
            },
        },
        {
            "id": "blackhole",
            "name": "Tenstorrent Blackhole p100",
            "memory_per_device_gb": 32,
            "devices_per_node": 8,
            "interconnect": "ethernet",
            "best_for": "Large-scale pre-training and 70B+ models",
            "available_nodes": 16,
            "pricing": {
                "on_demand": {"per_node_hour": "$6.80", "interruption_probability": None},
                "spot": {"per_node_hour": "$2.72", "interruption_probability": "~10%"},
            },
        },
    ],
    "queue": {
        "wormhole": {"estimated_wait_minutes": 2},
        "blackhole": {"estimated_wait_minutes": 8},
    },
}
