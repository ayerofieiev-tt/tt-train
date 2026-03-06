#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Shared constants for job_runner.py and session_worker.py."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Model catalog: tt:// path → HuggingFace repo + ttml model config
# ---------------------------------------------------------------------------

MODEL_CATALOG: dict[str, dict] = {
    "tt://catalog/llama-3.2-8b": {
        "hf_repo": "meta-llama/Llama-3.2-8B",
        "model_config": "model_configs/llama8b.yaml",
    },
    "tt://catalog/llama-3.2-1b": {
        "hf_repo": "meta-llama/Llama-3.2-1B-Instruct",
        "model_config": "model_configs/llama3_2_1B.yaml",
    },
    "tt://catalog/llama-3.1-70b": {
        "hf_repo": "meta-llama/Llama-3.1-70B",
        "model_config": "model_configs/llama8b.yaml",  # placeholder
    },
    "tt://catalog/tinyllama": {
        "hf_repo": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "model_config": "model_configs/tinyllama.yaml",
    },
    "tt://catalog/mistral-7b-v0.3": {
        "hf_repo": "mistralai/Mistral-7B-v0.3",
        "model_config": "model_configs/llama8b.yaml",  # same arch
    },
}

TT_METAL_HOME = Path("/home/boxx/tt-metal")
TTML_CONFIG_DIR = TT_METAL_HOME / "tt-train" / "configs"
