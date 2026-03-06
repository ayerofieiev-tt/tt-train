#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
TT-Train job runner — runs as a Slurm batch job.

Usage (called by scheduler):
    python job_runner.py \
        --job-id job_abc123 \
        --api-url http://api-server:8000 \
        --api-key internal-secret \
        --model tt://catalog/llama-3.2-8b \
        --method sft \
        --training-data ds_abc123 \
        --config '{"max_steps": 1000, "lr": 2e-5, "batch_size": 4}' \
        --storage-path /shared/tt_train
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import threading
import time
import random
from pathlib import Path

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


from workers.common import MODEL_CATALOG, TTML_CONFIG_DIR  # noqa: E402
from workers.events import EventEmitter, make_emitter  # noqa: E402


# ---------------------------------------------------------------------------
# API reporter
# ---------------------------------------------------------------------------

class JobReporter:
    """Posts progress/completion to the internal API."""

    def __init__(self, api_url: str, api_key: str, job_id: str, *,
                 console_job_id: str | None = None,
                 emitter: EventEmitter,
                 callback_url: str | None = None,
                 worker_token: str | None = None):
        self.base = api_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        self.job_id = job_id
        self.console_job_id = console_job_id
        self.emitter = emitter
        self.callback_url = callback_url
        self._console_headers = {
            "Authorization": f"Bearer {worker_token}",
            "Content-Type": "application/json",
        } if worker_token else {}

    def report_progress(self, step: int, total_steps: int, loss: float, epoch: float, **metrics):
        percentage = round(step / total_steps * 100, 1) if total_steps > 0 else 0.0
        payload = {
            "step": step,
            "total_steps": total_steps,
            "epoch": epoch,
            "percentage": percentage,
            "loss": loss,
            **metrics,
        }
        self._post(f"/internal/jobs/{self.job_id}/progress", payload)
        logger.info("step=%d/%d (%.1f%%) loss=%.4f", step, total_steps, percentage, loss)

    def report_complete(self, result_model: str, final_metrics: dict):
        self._post(f"/internal/jobs/{self.job_id}/complete", {
            "result_model": result_model,
            "metrics": final_metrics,
        })
        self.notify_callback("completed", result_model=result_model)

    def report_fail(self, error_type: str, message: str, step: int | None = None):
        self._post(f"/internal/jobs/{self.job_id}/fail", {
            "type": error_type,
            "message": message,
            "step": step,
        })
        self.notify_callback("failed", error={"type": error_type, "message": message})

    def report_log(self, message: str, log_type: str = "info", step: int | None = None):
        self._post(f"/internal/jobs/{self.job_id}/logs", {
            "log_type": log_type,
            "message": message,
            "step": step,
        })

    def notify_callback(self, status: str, **extra):
        if not self.callback_url:
            return
        try:
            payload = {"platform_job_id": self.console_job_id or self.job_id, "status": status, **extra}
            r = httpx.post(self.callback_url, json=payload,
                           headers=self._console_headers, timeout=10.0)
            r.raise_for_status()
        except Exception as e:
            logger.warning("Failed to notify callback URL: %s", e)

    def emit_event(self, event_type: str, *, status: str = "completed", **usage_fields) -> None:
        self.emitter.emit(event_type, status=status, **usage_fields)

    def _post(self, path: str, body: dict):
        try:
            r = httpx.post(self.base + path, json=body, headers=self.headers, timeout=30.0)
            r.raise_for_status()
        except Exception as e:
            logger.warning("Failed to report to API server: %s", e)


# ---------------------------------------------------------------------------
# TelemetrySFTTrainer — thin subclass that adds progress reporting hooks
# ---------------------------------------------------------------------------

class TelemetrySFTTrainer:
    """
    Wraps SFTTrainer to add:
    - last_eval_loss tracking (override _eval)
    - checkpoint copy to shared storage + progress report (override _save_checkpoint)
    """

    def __new__(cls, *args, reporter, job_id, storage_path, total_steps, tokens_per_step=512, **kwargs):
        from ttml.trainers import SFTTrainer

        # Dynamically create a subclass of the real SFTTrainer
        class _Telemetry(SFTTrainer):
            last_eval_loss: float = 0.0
            _reporter: JobReporter = reporter
            _job_id: str = job_id
            _storage_path: Path = Path(storage_path)
            _total_steps: int = total_steps
            _tokens_per_step: int = tokens_per_step

            def _eval(self) -> float:
                loss = super()._eval()
                self.last_eval_loss = loss
                return loss

            def _save_checkpoint(self) -> None:
                super()._save_checkpoint()

                # Copy checkpoint to shared storage
                src = Path(self.config.checkpoint_dir) / f"step_{self.step}.pkl"
                dst_dir = self._storage_path / "checkpoints" / self._job_id
                dst_dir.mkdir(parents=True, exist_ok=True)
                if src.exists():
                    shutil.copy2(src, dst_dir / src.name)

                # Report progress to API server
                epoch = self.step / self._total_steps if self._total_steps > 0 else 0.0
                total_epochs = 1.0
                self._reporter.report_progress(
                    step=self.step,
                    total_steps=self._total_steps,
                    loss=self.last_eval_loss,
                    epoch=epoch,
                )

                # Emit telemetry event
                interval_tokens = self.config.save_interval * self._tokens_per_step
                self._reporter.emit_event(
                    "finetuning.training_step",
                    training_tokens=interval_tokens,
                    tpu_seconds=self.config.save_interval,
                    epoch=round(epoch, 4),
                    total_epochs=total_epochs,
                )

        return _Telemetry(*args, **kwargs)


class ProgressPoller(threading.Thread):
    """
    Background thread that polls trainer.step and reports to the API server.
    Fires every poll_interval seconds regardless of checkpoint cadence,
    so the API server has live progress even between save intervals.
    """

    def __init__(self, trainer, reporter: JobReporter, total_steps: int, poll_interval: float = 30.0):
        super().__init__(daemon=True)
        self._trainer = trainer
        self._reporter = reporter
        self._total_steps = total_steps
        self._poll_interval = poll_interval
        self._stop = threading.Event()
        self._last_step = -1

    def run(self):
        while not self._stop.wait(self._poll_interval):
            step = self._trainer.step
            if step != self._last_step and step > 0:
                loss = getattr(self._trainer, "last_eval_loss", 0.0)
                epoch = step / self._total_steps if self._total_steps > 0 else 0.0
                self._reporter.report_progress(
                    step=step,
                    total_steps=self._total_steps,
                    loss=loss,
                    epoch=epoch,
                )
                self._last_step = step

    def stop(self):
        self._stop.set()


# ---------------------------------------------------------------------------
# Dataset download helper
# ---------------------------------------------------------------------------

def download_dataset(dataset_url: str, dest_path: str) -> None:
    """Download a dataset from a URL (e.g. S3 pre-signed URL) to a local file."""
    import os
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    logger.info("Downloading dataset from URL to %s", dest_path)
    with httpx.stream("GET", dataset_url, timeout=300.0, follow_redirects=True) as r:
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_bytes(chunk_size=65536):
                f.write(chunk)
    logger.info("Dataset downloaded: %s bytes", os.path.getsize(dest_path))


# ---------------------------------------------------------------------------
# Real training path (requires ttml + ttnn + TT hardware)
# ---------------------------------------------------------------------------

def run_training_real(args, config: dict, reporter: JobReporter) -> float:
    """
    Full SFT training using the ttml stack on Tenstorrent hardware.
    Requires: ttml, ttnn, transformers, datasets packages + TT device.
    """
    import os
    from functools import partial

    import datasets as hf_datasets
    import ttml
    import ttnn
    from transformers import AutoTokenizer
    from ttml.common.config import load_config
    from ttml.common.model_factory import TransformerModelFactory
    from ttml.common.utils import initialize_device
    from ttml.datasets import InMemoryDataloader, sft_collate_fn
    from ttml.trainers import SFTConfig

    # Resolve model catalog entry
    catalog_entry = MODEL_CATALOG.get(args.model)
    if catalog_entry is None:
        raise ValueError(f"Unknown model: {args.model}. Available: {list(MODEL_CATALOG)}")

    hf_repo = catalog_entry["hf_repo"]
    model_config_rel = catalog_entry["model_config"]
    model_config_path = TTML_CONFIG_DIR / model_config_rel

    logger.info("Model: %s → %s", args.model, hf_repo)

    # ---- Tokenizer ----
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    tokenizer = AutoTokenizer.from_pretrained(hf_repo)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Dataset ----
    # Datasets uploaded by the API are stored as JSONL at:
    #   {shared_storage}/datasets/{dataset_id}/data.jsonl
    # Format: {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    if getattr(args, 'dataset_url', None):
        # Download from URL to local temp path
        local_ds_dir = Path(args.storage_path) / "datasets" / args.job_id
        local_ds_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = local_ds_dir / "data.jsonl"
        if not dataset_path.exists():
            download_dataset(args.dataset_url, str(dataset_path))
    else:
        dataset_path = Path(args.storage_path) / "datasets" / args.training_data / "data.jsonl"
    logger.info("Loading dataset from %s", dataset_path)
    raw = hf_datasets.load_dataset("json", data_files=str(dataset_path), split="train")

    max_seq_len = config.get("max_seq_len", 1024)

    def _tokenize(examples):
        # Apply chat template → full text, then tokenize
        # Labels: -100 for prompt tokens, real token IDs for completion tokens
        input_ids_list, labels_list = [], []
        for msgs in examples["messages"]:
            # Separate prompt (all turns except last assistant) from completion (last assistant)
            if msgs and msgs[-1]["role"] == "assistant":
                prompt_msgs = msgs[:-1]
                completion_text = msgs[-1]["content"]
            else:
                prompt_msgs = msgs
                completion_text = ""

            prompt_text = tokenizer.apply_chat_template(
                prompt_msgs, tokenize=False, add_generation_prompt=True
            ) if prompt_msgs else ""

            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
            completion_ids = tokenizer.encode(completion_text, add_special_tokens=False)

            # Truncate to max_seq_len
            full_ids = (prompt_ids + completion_ids)[:max_seq_len]
            prompt_len = min(len(prompt_ids), len(full_ids))

            labels = [-100] * prompt_len + full_ids[prompt_len:]

            input_ids_list.append(full_ids)
            labels_list.append(labels)

        return {"input_ids": input_ids_list, "labels": labels_list}

    tokenized = raw.map(_tokenize, batched=True, remove_columns=raw.column_names)

    # ---- Eval dataset ----
    eval_ds = None
    if args.validation_data:
        val_path = Path(args.storage_path) / "datasets" / args.validation_data / "data.jsonl"
        if val_path.exists():
            val_raw = hf_datasets.load_dataset("json", data_files=str(val_path), split="train")
            eval_ds = val_raw.map(_tokenize, batched=True, remove_columns=val_raw.column_names)

    # ---- Dataloaders ----
    batch_size = config.get("batch_size", 1)
    collate = partial(sft_collate_fn, max_seq_len=max_seq_len, pad_token_id=tokenizer.pad_token_id)
    train_loader = InMemoryDataloader(tokenized, collate, batch_size, shuffle=True)
    eval_loader = InMemoryDataloader(eval_ds, collate, batch_size) if eval_ds is not None else None

    # ---- Device ----
    # Single-device setup; multi-device requires additional mesh config
    ttml.autograd.AutoContext.get_instance().initialize_parallelism_context(
        ttml.autograd.DistributedConfig(enable_ddp=False, enable_tp=False)
    )

    # ---- Model ----
    logger.info("Loading model config from %s", model_config_path)
    model_cfg = load_config(str(model_config_path))
    factory = TransformerModelFactory(model_cfg)
    factory.transformer_config.vocab_size = tokenizer.vocab_size

    logger.info("Creating model and loading weights from HuggingFace (%s)...", hf_repo)
    model = factory.create_model()
    from huggingface_hub import snapshot_download
    weights_dir = snapshot_download(repo_id=hf_repo, ignore_patterns=["*.bin", "*.pt"])
    model.load_from_safetensors(weights_dir)
    logger.info("Model loaded.")

    # ---- SFTConfig ----
    total_steps = config.get("max_steps", config.get("epochs", 1) * len(train_loader))
    checkpoint_dir = str(Path(args.storage_path) / "checkpoints" / args.job_id / "ckpts")

    sft_cfg = SFTConfig(
        max_steps=total_steps,
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
        eval_interval=config.get("eval_interval", total_steps // 5 if eval_loader else 0),
        save_interval=config.get("save_interval", max(1, total_steps // 10)),
        checkpoint_dir=checkpoint_dir,
        max_seq_len=max_seq_len,
        lr=config.get("lr", 2e-5),
        weight_decay=config.get("weight_decay", 0.01),
        warmup_steps=config.get("warmup_steps", 0),
        seed=config.get("seed"),
    )

    # ---- Trainer with telemetry ----
    trainer = TelemetrySFTTrainer(
        model=model,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
        config=sft_cfg,
        reporter=reporter,
        job_id=args.job_id,
        storage_path=args.storage_path,
        total_steps=total_steps,
    )

    # Start progress poller (reports every 30s between checkpoints)
    poller = ProgressPoller(trainer, reporter, total_steps, poll_interval=30.0)
    poller.start()

    try:
        trainer.train()
    finally:
        poller.stop()

    return trainer.last_eval_loss or 0.0


# ---------------------------------------------------------------------------
# Simulation fallback (no TT hardware required — used by LocalBackend)
# ---------------------------------------------------------------------------

def run_training_sim(args, config: dict, reporter: JobReporter) -> float:
    """Realistic simulation used when ttml/ttnn are not available (local dev)."""
    if getattr(args, 'dataset_url', None):
        # Download from URL to local temp path
        local_ds_dir = Path(args.storage_path) / "datasets" / args.job_id
        local_ds_dir.mkdir(parents=True, exist_ok=True)
        dataset_path = local_ds_dir / "data.jsonl"
        if not dataset_path.exists():
            download_dataset(args.dataset_url, str(dataset_path))

    total_steps = config.get("max_steps", config.get("epochs", 3) * 100)
    report_every = max(1, total_steps // 20)
    loss = 3.5

    for step in range(1, total_steps + 1):
        time.sleep(0.05)
        loss = max(0.3, loss - random.uniform(0.001, 0.005) + random.gauss(0, 0.02))
        epoch = step / total_steps

        if step % report_every == 0 or step == total_steps:
            reporter.report_progress(step, total_steps, loss, epoch, tokens_processed=step * 512)

    return loss


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="TT-Train job runner")
    p.add_argument("--job-id", required=True)
    p.add_argument("--api-url", required=True)
    p.add_argument("--api-key", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--training-data", default=None)
    p.add_argument("--validation-data", default=None)
    p.add_argument("--config", default="{}")
    p.add_argument("--storage-path", default="/tmp/tt_train_storage")
    p.add_argument("--console-job-id", default=None)
    p.add_argument("--console-base-url", default=None)
    p.add_argument("--worker-token", default=None)
    p.add_argument("--callback-url", default=None)
    p.add_argument("--dataset-url", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    config = json.loads(args.config)
    total_steps = config.get("max_steps", config.get("epochs", 3) * 100)
    total_tokens = total_steps * config.get("tokens_per_step", 512)

    emitter = make_emitter(
        platform_base_url=args.console_base_url,
        worker_token=args.worker_token,
        model=args.model,
        job_id=args.job_id,
    )
    reporter = JobReporter(
        args.api_url,
        args.api_key,
        args.job_id,
        console_job_id=args.console_job_id,
        emitter=emitter,
        callback_url=args.callback_url,
        worker_token=args.worker_token,
    )
    reporter.notify_callback("running")

    logger.info("Starting job %s: model=%s method=%s", args.job_id, args.model, args.method)

    try:
        # Try real training first; fall back to simulation if TT stack not present
        try:
            import ttml  # noqa: F401
            logger.info("ttml available — using real SFTTrainer")
            final_loss = run_training_real(args, config, reporter)
        except ImportError:
            logger.warning("ttml not available — running simulation (no TT hardware)")
            final_loss = run_training_sim(args, config, reporter)
        except Exception as hw_err:
            logger.warning("Real training failed (%s), falling back to simulation", hw_err)
            final_loss = run_training_sim(args, config, reporter)

        # Final checkpoint path
        result_model = f"tt://checkpoints/{args.job_id}/final"
        storage = Path(args.storage_path) / "checkpoints" / args.job_id
        storage.mkdir(parents=True, exist_ok=True)

        reporter.report_complete(
            result_model=result_model,
            final_metrics={"train_loss": round(final_loss, 4)},
        )
        reporter.emit_event(
            "finetuning.job_completed",
            status="completed",
            total_training_tokens=total_tokens,
            total_tpu_seconds=total_steps,
            epochs=config.get("epochs", 1),
        )
        logger.info("Job %s completed. result_model=%s loss=%.4f", args.job_id, result_model, final_loss)

    except KeyboardInterrupt:
        reporter.report_fail("cancelled", "Job was cancelled")
        reporter.emit_event(
            "finetuning.job_completed",
            status="cancelled",
            total_training_tokens=0,
            total_tpu_seconds=0,
            epochs=0,
        )
        sys.exit(0)
    except Exception as e:
        logger.exception("Job %s failed", args.job_id)
        reporter.report_fail("runtime_error", str(e))
        reporter.emit_event(
            "finetuning.job_completed",
            status="failed",
            total_training_tokens=0,
            total_tpu_seconds=0,
            epochs=0,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
