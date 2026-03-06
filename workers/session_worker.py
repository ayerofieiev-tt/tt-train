#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
TT-Train session worker — runs as a long-running Slurm job.

Exposes a local HTTP server that the API server proxies commands to.
Registers with the API server via POST /internal/sessions/{id}/ready on startup.

When ttml/ttnn are available (TT hardware present), uses InteractiveSFTTrainer —
a thin subclass of SFTTrainer that exposes forward_backward / step / sample /
log_probs / eval / save as individual callable methods instead of running the
full autonomous .train() loop.

Falls back to SimModelState when TT hardware is not present (local dev).
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import socket
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from threading import Event, Lock, Thread

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

from workers.common import MODEL_CATALOG, TTML_CONFIG_DIR


# ---------------------------------------------------------------------------
# InteractiveSFTTrainer
# ---------------------------------------------------------------------------

class InteractiveSFTTrainer:
    """
    Thin subclass of SFTTrainer that exposes the training loop's individual
    pieces as callable methods, enabling user-driven (session) training.

    SFTTrainer.__init__ sets up everything we need:
        self._optimizer      AdamW
        self._lr_schedule    step -> lr callable
        self._causal_mask    [1,1,T,T] bfloat16 tensor
        self._loss_fn        cross_entropy_loss

    We skip calling .train() entirely. Instead the session worker calls
    forward_backward() / step() / sample() / log_probs() / eval() / save()
    in whatever order and cadence the user drives from the SDK.

    Gradient accumulation works naturally: forward_backward() accumulates
    grads, step() applies and clears them. The user controls how many
    forward_backward() calls happen before each step().
    """

    def __new__(cls, model, config, lr_schedule=None):
        from ttml.trainers import SFTTrainer

        class _Interactive(SFTTrainer):
            # Track whether we need to zero_grad before the next forward pass.
            # True at init and after each step().
            _needs_zero_grad: bool = True

            def forward_backward(self, batch, loss_fn: str = "cross_entropy") -> dict:
                """
                Compute loss on one batch and accumulate gradients.
                Gradients persist until step() is called.
                Multiple forward_backward() calls before step() = gradient accumulation.
                """
                import time as _time
                import numpy as np
                import ttnn
                import ttml

                t0 = _time.monotonic()

                if self._needs_zero_grad:
                    self._optimizer.zero_grad()
                    self._needs_zero_grad = False

                loss_tensor = self._compute_loss(batch)
                loss_value = float(loss_tensor.to_numpy(ttnn.DataType.FLOAT32).mean())

                loss_tensor.backward(False)
                ttml.autograd.AutoContext.get_instance().reset_graph()

                duration_ms = int((_time.monotonic() - t0) * 1000)

                # token_count: batch shape is [B,1,1,T] for input_ids
                token_count = int(np.prod(batch.input_ids.shape[:1])) * self.config.max_seq_len

                return {
                    "loss": round(loss_value, 6),
                    "token_count": token_count,
                    "example_count": batch.input_ids.shape[0],
                    "grad_norm": None,   # computed after step() in some frameworks
                    "duration_ms": duration_ms,
                }

            def optimizer_step(self, max_grad_norm: float | None = None) -> dict:
                """
                Apply accumulated gradients, advance the LR schedule, reset grads.
                """
                import time as _time
                t0 = _time.monotonic()

                lr = self._lr_schedule(self.step)
                self._optimizer.set_lr(lr)
                self._optimizer.step()
                self.step += 1
                self._needs_zero_grad = True

                duration_ms = int((_time.monotonic() - t0) * 1000)
                return {
                    "step_number": self.step,
                    "learning_rate": lr,
                    "grad_norm_before_clip": None,
                    "grad_norm_after_clip": None,
                    "duration_ms": duration_ms,
                }

            def sample(
                self,
                prompts: list[list[int]],
                tokenizer,
                max_tokens: int = 256,
                temperature: float = 1.0,
                top_p: float = 1.0,
                stop_ids: list[int] | None = None,
                return_log_probs: bool = False,
                seed: int | None = None,
            ) -> list[dict]:
                """
                Generate one completion per prompt (list of token-id lists).
                Uses greedy/temperature sampling via ttml.ops.sample.sample_op.
                """
                import numpy as np
                import ttnn
                import ttml
                from ttml.common.utils import no_grad

                if seed is not None:
                    ttml.autograd.AutoContext.get_instance().set_seed(seed)

                eos_id = tokenizer.eos_token_id
                pad_id = tokenizer.pad_token_id or eos_id
                device = ttml.autograd.AutoContext.get_instance().get_device()
                composer = ttml.core.distributed.concat_mesh_to_tensor_composer(device, 0)
                max_seq = self.config.max_seq_len

                self.model.eval()
                results = []

                with no_grad():
                    for prompt_ids in prompts:
                        generated: list[int] = []
                        current = list(prompt_ids)

                        for _ in range(max_tokens):
                            # Build padded input [1,1,1,T]
                            window = current[-max_seq:]
                            padded = np.full((1, 1, 1, max_seq), pad_id, dtype=np.uint32)
                            padded[0, 0, 0, :len(window)] = window

                            input_tensor = ttml.autograd.Tensor.from_numpy(
                                padded, ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32
                            )

                            logits = self.model(input_tensor, self._causal_mask)

                            rng = seed if seed is not None else np.random.randint(1, int(1e7))
                            next_tok_tensor = ttml.ops.sample.sample_op(
                                logits, temperature, rng, None
                            )

                            next_tok = int(
                                next_tok_tensor.to_numpy(composer=composer).reshape(-1)[
                                    len(window) - 1
                                ]
                            )

                            if next_tok == eos_id or (stop_ids and next_tok in stop_ids):
                                break

                            generated.append(next_tok)
                            current.append(next_tok)

                        result = {
                            "text": tokenizer.decode(generated, skip_special_tokens=True),
                            "tokens": len(generated),
                            "finish_reason": "stop" if len(generated) < max_tokens else "length",
                        }
                        if return_log_probs:
                            result["log_prob"] = None   # full log-prob scoring via log_probs()
                            result["per_token_log_probs"] = []
                        results.append(result)

                self.model.train()
                return results

            def score_log_probs(self, batches: list) -> list[dict]:
                """
                Forward-pass only (no grad) on completion batches.
                batches: list of Batch objects (one per {prompt, completion} pair).
                Returns per-batch log-prob scores.
                """
                import numpy as np
                import ttnn
                import ttml
                from ttml.common.utils import no_grad

                scores = []
                self.model.eval()
                with no_grad():
                    for i, batch in enumerate(batches):
                        loss_tensor = self._compute_loss(batch)
                        loss_val = float(loss_tensor.to_numpy(ttnn.DataType.FLOAT32).mean())
                        # Convert mean loss to total log-prob (neg log-prob = loss)
                        token_count = int(np.prod(batch.labels.shape))
                        total_lp = -loss_val * token_count
                        scores.append({
                            "index": i,
                            "total_log_prob": round(total_lp, 4),
                            "avg_log_prob": round(-loss_val, 4),
                            "tokens": token_count,
                            "per_token": [],
                        })
                self.model.train()
                return scores

            def eval_loss(self, dataloader) -> dict:
                """Run eval pass and return loss + perplexity."""
                import math
                import time as _time
                t0 = _time.monotonic()
                val_loss = self._eval_with_loader(dataloader)
                duration_ms = int((_time.monotonic() - t0) * 1000)
                return {
                    "loss": round(val_loss, 4),
                    "perplexity": round(math.exp(val_loss), 4),
                    "duration_ms": duration_ms,
                }

            def _eval_with_loader(self, dataloader) -> float:
                """Like _eval() but takes an explicit dataloader instead of self.eval_dataloader."""
                import numpy as np
                import ttnn
                from ttml.common.utils import no_grad

                self.model.eval()
                losses = []
                with no_grad():
                    for batch in dataloader:
                        loss = self._compute_loss(batch)
                        losses.append(float(loss.to_numpy(ttnn.DataType.FLOAT32).mean()))
                self.model.train()
                return float(np.mean(losses)) if losses else 0.0

            def save_checkpoint(self, checkpoint_dir: str, step: int | None = None) -> str:
                """Save a checkpoint and return the file path."""
                import os
                step_num = step if step is not None else self.step
                os.makedirs(checkpoint_dir, exist_ok=True)
                # Reuse parent's _save_checkpoint by temporarily pointing config.checkpoint_dir
                original_dir = self.config.checkpoint_dir
                self.config.checkpoint_dir = checkpoint_dir
                self._save_checkpoint()
                self.config.checkpoint_dir = original_dir
                return str(Path(checkpoint_dir) / f"step_{step_num}.pkl")

        return _Interactive(
            model=model,
            train_dataloader=[],   # not used; we never call .train()
            eval_dataloader=None,
            config=config,
            lr_schedule=lr_schedule,
        )


# ---------------------------------------------------------------------------
# RealModelState — wraps InteractiveSFTTrainer + tokenizer
# ---------------------------------------------------------------------------

class RealModelState:
    """
    Bridges the JSON API surface (messages, batches as dicts) and the ttml
    tensor world (Batch objects, tokenized ids).
    """

    def __init__(
        self,
        model_id: str,
        optimizer_config: dict,
        storage_path: str,
        session_id: str,
    ):
        import os
        from functools import partial

        import ttml
        import ttnn
        from transformers import AutoTokenizer
        from ttml.common.config import load_config
        from ttml.common.model_factory import TransformerModelFactory
        from ttml.common.utils import initialize_device
        from ttml.datasets import sft_collate_fn
        from ttml.trainers import SFTConfig

        catalog_entry = MODEL_CATALOG.get(model_id)
        if catalog_entry is None:
            raise ValueError(f"Unknown model: {model_id}")

        hf_repo = catalog_entry["hf_repo"]
        model_config_path = TTML_CONFIG_DIR / catalog_entry["model_config"]

        # Tokenizer
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        self.tokenizer = AutoTokenizer.from_pretrained(hf_repo)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Device (single)
        ttml.autograd.AutoContext.get_instance().initialize_parallelism_context(
            ttml.autograd.DistributedConfig(enable_ddp=False, enable_tp=False)
        )

        # Model
        logger.info("Loading model %s from %s...", model_id, hf_repo)
        model_cfg = load_config(str(model_config_path))
        factory = TransformerModelFactory(model_cfg)
        factory.transformer_config.vocab_size = self.tokenizer.vocab_size
        model = factory.create_model()
        from huggingface_hub import snapshot_download
        weights_dir = snapshot_download(repo_id=hf_repo, ignore_patterns=["*.bin", "*.pt"])
        model.load_from_safetensors(weights_dir)
        logger.info("Model loaded.")

        self.max_seq_len = factory.transformer_config.max_sequence_length
        self._collate = partial(
            sft_collate_fn,
            max_seq_len=self.max_seq_len,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # SFTConfig (drives optimizer + LR schedule inside the trainer)
        lr = optimizer_config.get("lr", 2e-5)
        sft_cfg = SFTConfig(
            max_steps=999_999,   # unbounded — user controls how many steps
            lr=lr,
            weight_decay=optimizer_config.get("weight_decay", 0.01),
            max_seq_len=self.max_seq_len,
            save_interval=0,     # user triggers saves manually
            eval_interval=0,     # user triggers evals manually
            checkpoint_dir=str(Path(storage_path) / "checkpoints" / session_id / "ckpts"),
        )

        self.trainer = InteractiveSFTTrainer(model, sft_cfg)
        self.step_count = 0
        self._storage_path = storage_path
        self._session_id = session_id
        self._last_loss = 0.0

    # ---- helpers ----

    def _messages_to_batch(self, messages_batch: list[list[dict]]):
        """Convert a list of messages-format examples into a ttml Batch."""
        examples = []
        for msgs in messages_batch:
            if msgs and msgs[-1]["role"] == "assistant":
                prompt_msgs, completion = msgs[:-1], msgs[-1]["content"]
            else:
                prompt_msgs, completion = msgs, ""

            prompt_text = (
                self.tokenizer.apply_chat_template(
                    prompt_msgs, tokenize=False, add_generation_prompt=True
                )
                if prompt_msgs else ""
            )
            prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
            completion_ids = self.tokenizer.encode(completion, add_special_tokens=False)

            full_ids = (prompt_ids + completion_ids)[:self.max_seq_len]
            prompt_len = min(len(prompt_ids), len(full_ids))
            labels = [-100] * prompt_len + full_ids[prompt_len:]

            examples.append({"input_ids": full_ids, "labels": labels})

        return self._collate(examples)

    def _normalize_prompt(self, prompt) -> list[int]:
        """Convert string or messages dict to token IDs."""
        if isinstance(prompt, str):
            return self.tokenizer.encode(prompt, add_special_tokens=False)
        msgs = prompt.get("messages", [prompt]) if isinstance(prompt, dict) else prompt
        text = self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        return self.tokenizer.encode(text, add_special_tokens=False)

    # ---- API surface (matches SimModelState) ----

    def forward_backward(self, batch: list, loss_fn: str, loss_config: dict) -> dict:
        ttml_batch = self._messages_to_batch(
            [ex.get("messages", []) for ex in batch]
        )
        result = self.trainer.forward_backward(ttml_batch, loss_fn)
        self._last_loss = result["loss"]
        return result

    def step(self, max_grad_norm: float | None) -> dict:
        result = self.trainer.optimizer_step(max_grad_norm)
        self.step_count = result["step_number"]
        return result

    def sample(self, prompts, temperature, top_p, max_tokens, n, return_log_probs, **kwargs) -> dict:
        completions = []
        for i, p in enumerate(prompts):
            prompt_ids = self._normalize_prompt(p)
            outputs = []
            for j in range(n):
                results = self.trainer.sample(
                    [prompt_ids],
                    self.tokenizer,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    return_log_probs=return_log_probs,
                    seed=kwargs.get("seed"),
                )
                out = results[0]
                outputs.append({"index": j, **out})
            completions.append({"prompt_index": i, "outputs": outputs})

        total_tokens = sum(o["tokens"] for c in completions for o in c["outputs"])
        return {
            "completions": completions,
            "usage": {"prompt_tokens": sum(len(self._normalize_prompt(p)) for p in prompts), "completion_tokens": total_tokens},
        }

    def log_probs(self, batch: list) -> dict:
        ttml_batches = [
            self._messages_to_batch([
                [
                    {"role": "user", "content": ex.get("prompt", "")},
                    {"role": "assistant", "content": ex.get("completion", "")},
                ]
            ])
            for ex in batch
        ]
        scores = self.trainer.score_log_probs(ttml_batches)
        return {"scores": scores}

    def eval(self, data, metrics: list | None, max_examples: int | None, batch_size: int) -> dict:
        from functools import partial
        from ttml.datasets import InMemoryDataloader

        if isinstance(data, str):
            # data is a dataset ID — load from shared storage
            import datasets as hf_datasets
            path = Path(self._storage_path) / "datasets" / data / "data.jsonl"
            raw = hf_datasets.load_dataset("json", data_files=str(path), split="train")
            if max_examples:
                raw = raw.select(range(min(max_examples, len(raw))))
            examples = [{"input_ids": self._normalize_prompt(r), "labels": self._normalize_prompt(r)} for r in raw]
        else:
            examples = []
            for ex in (data[:max_examples] if max_examples else data):
                msgs = ex.get("messages", [])
                ttml_batch_list = [msgs]
                examples.append({"input_ids": self._normalize_prompt(msgs), "labels": self._normalize_prompt(msgs)})

        loader = InMemoryDataloader(examples, self._collate, batch_size=batch_size)
        result = self.trainer.eval_loss(loader)

        requested = set(metrics or ["loss", "perplexity"])
        return {
            "examples_evaluated": len(examples),
            "metrics": {k: v for k, v in result.items() if k in requested},
            "duration_ms": result.get("duration_ms", 0),
        }

    def save(self, name: str | None, metadata: dict, storage_path: str, session_id: str) -> dict:
        import uuid
        ckpt_id = f"ckpt_{session_id[-8:]}_{self.step_count:06d}"
        ckpt_dir = str(Path(storage_path) / "checkpoints" / session_id / ckpt_id)
        pkl_path = self.trainer.save_checkpoint(ckpt_dir, self.step_count)
        model_path = f"tt://checkpoints/{session_id}/{ckpt_id}"
        return {
            "id": ckpt_id,
            "model_path": model_path,
            "session_id": session_id,
            "step": self.step_count,
            "name": name,
            "metrics": {"train_loss": round(self._last_loss, 4)},
            "metadata": metadata or {},
        }

    @property
    def step_count_prop(self) -> int:
        return self.trainer.step


# ---------------------------------------------------------------------------
# SimModelState — simulation fallback (no TT hardware)
# ---------------------------------------------------------------------------

class SimModelState:
    """Simulation used when ttml/ttnn are not available (local dev)."""

    def __init__(self, model_id: str, lora_config: dict, optimizer_config: dict):
        self.model_id = model_id
        self.lora_config = lora_config
        self.optimizer_config = optimizer_config
        self.step_count = 0
        self._loss = 3.5
        logger.info("Simulation mode: model %s (no TT hardware)", model_id)

    def forward_backward(self, batch: list, loss_fn: str, loss_config: dict) -> dict:
        n = len(batch)
        self._loss = max(0.3, self._loss - random.uniform(0.001, 0.004) + random.gauss(0, 0.015))
        return {
            "loss": round(self._loss, 4),
            "token_count": n * random.randint(40, 120),
            "example_count": n,
            "grad_norm": round(random.uniform(0.1, 2.0), 4),
            "duration_ms": random.randint(200, 500),
        }

    def step(self, max_grad_norm: float | None) -> dict:
        self.step_count += 1
        lr = self.optimizer_config.get("lr", 2e-5)
        grad_norm = round(random.uniform(0.1, 1.5), 4)
        clipped = min(grad_norm, max_grad_norm or grad_norm)
        return {
            "step_number": self.step_count,
            "learning_rate": lr,
            "grad_norm_before_clip": grad_norm,
            "grad_norm_after_clip": round(clipped, 4),
            "duration_ms": random.randint(10, 30),
        }

    def sample(self, prompts, temperature, top_p, max_tokens, n, return_log_probs, **kwargs) -> dict:
        responses = [
            "That's a great question! Here's what I think...",
            "The answer depends on several factors.",
            "Based on the context, I would suggest...",
        ]
        completions = []
        for i, _ in enumerate(prompts):
            outputs = []
            for j in range(n):
                text = random.choice(responses)
                out = {"index": j, "text": text, "tokens": len(text.split()), "finish_reason": "stop"}
                if return_log_probs:
                    out["log_prob"] = round(random.uniform(-3.0, -0.5), 4)
                    out["per_token_log_probs"] = []
                outputs.append(out)
            completions.append({"prompt_index": i, "outputs": outputs})
        total = sum(o["tokens"] for c in completions for o in c["outputs"])
        return {
            "completions": completions,
            "usage": {"prompt_tokens": len(prompts) * 20, "completion_tokens": total},
        }

    def log_probs(self, batch: list) -> dict:
        return {
            "scores": [
                {
                    "index": i,
                    "total_log_prob": round(random.uniform(-10.0, -1.0), 4),
                    "avg_log_prob": round(random.uniform(-3.0, -0.3), 4),
                    "tokens": len(str(ex.get("completion", "")).split()),
                    "per_token": [],
                }
                for i, ex in enumerate(batch)
            ]
        }

    def eval(self, data, metrics: list | None, max_examples: int | None, batch_size: int) -> dict:
        requested = set(metrics or ["loss", "perplexity", "accuracy"])
        all_metrics = {
            "loss": round(self._loss * 1.1, 4),
            "perplexity": round(2 ** (self._loss * 1.1), 2),
            "accuracy": round(random.uniform(0.6, 0.95), 4),
        }
        return {
            "examples_evaluated": max_examples or 100,
            "metrics": {k: v for k, v in all_metrics.items() if k in requested},
            "duration_ms": random.randint(500, 2000),
        }

    def save(self, name: str | None, metadata: dict, storage_path: str, session_id: str) -> dict:
        ckpt_id = f"ckpt_{session_id[-8:]}_{self.step_count:06d}"
        Path(storage_path, "checkpoints", session_id).mkdir(parents=True, exist_ok=True)
        return {
            "id": ckpt_id,
            "model_path": f"tt://checkpoints/{session_id}/{ckpt_id}",
            "session_id": session_id,
            "step": self.step_count,
            "name": name,
            "metrics": {"train_loss": round(self._loss, 4)},
            "metadata": metadata or {},
        }


# ---------------------------------------------------------------------------
# RequestStore — tracks in-flight and completed async requests
# ---------------------------------------------------------------------------

class RequestStore:
    """Thread-safe store for async compute requests.

    submit() reserves a slot and returns a request_id.
    complete() stores the result and wakes any waiting retrieve callers.
    wait() long-polls for up to `timeout` seconds, returning the result or None.
    """

    def __init__(self):
        self._lock = Lock()
        self._events: dict[str, Event] = {}
        self._results: dict[str, dict] = {}

    def submit(self) -> str:
        request_id = str(uuid.uuid4())
        with self._lock:
            self._events[request_id] = Event()
        return request_id

    def complete(self, request_id: str, result: dict) -> None:
        with self._lock:
            self._results[request_id] = result
            event = self._events.get(request_id)
        if event:
            event.set()

    def wait(self, request_id: str, timeout: float = 45.0) -> dict | None:
        """Block up to `timeout` seconds. Returns result dict or None (not ready)."""
        with self._lock:
            result = self._results.get(request_id)
            event = self._events.get(request_id)
        if result is not None:
            return result
        if event is None:
            return {"error": f"unknown request_id: {request_id}"}
        ready = event.wait(timeout=timeout)
        if ready:
            with self._lock:
                return self._results.get(request_id)
        return None  # caller should retry


# ---------------------------------------------------------------------------
# Threaded HTTP server — allows concurrent retrieve calls while computing
# ---------------------------------------------------------------------------

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class WorkerHandler(BaseHTTPRequestHandler):
    model = None
    session_id: str = None
    storage_path: str = None
    shutdown_flag: list = None
    _store: RequestStore = None
    _executor: ThreadPoolExecutor = None  # max_workers=1: serial computation

    def log_message(self, fmt, *args):
        logger.debug("HTTP %s", fmt % args)

    # ------------------------------------------------------------------
    # Async submission helpers
    # ------------------------------------------------------------------

    def _submit(self, fn) -> str:
        """Enqueue fn on the serial executor, return request_id immediately."""
        request_id = self._store.submit()
        self._executor.submit(self._run, request_id, fn)
        return request_id

    def _run(self, request_id: str, fn):
        try:
            result = fn()
            self._store.complete(request_id, result)
        except Exception as e:
            logger.exception("Async command failed for request %s", request_id)
            self._store.complete(request_id, {"error": str(e)})

    # ------------------------------------------------------------------
    # Request dispatch
    # ------------------------------------------------------------------

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}
        path = self.path.rstrip("/")

        try:
            if path == "/forward_backward":
                sid, model, b = self.session_id, self.model, body
                rid = self._submit(lambda: {
                    "object": "forward_backward_result", "session_id": sid,
                    **model.forward_backward(
                        b.get("batch", []),
                        b.get("loss", "cross_entropy"),
                        b.get("loss_config") or {},
                    ),
                })
                self._respond(200, {"request_id": rid})

            elif path == "/step":
                sid, model, b = self.session_id, self.model, body
                rid = self._submit(lambda: {
                    "object": "step_result", "session_id": sid,
                    **model.step(b.get("max_grad_norm")),
                })
                self._respond(200, {"request_id": rid})

            elif path == "/sample":
                sid, model, b = self.session_id, self.model, body
                rid = self._submit(lambda: {
                    "object": "sample_result", "session_id": sid,
                    **model.sample(
                        b.get("prompts", []),
                        b.get("temperature", 1.0),
                        b.get("top_p", 1.0),
                        b.get("max_tokens", 256),
                        b.get("n", 1),
                        b.get("return_log_probs", False),
                        seed=b.get("seed"),
                    ),
                })
                self._respond(200, {"request_id": rid})

            elif path == "/log_probs":
                sid, model, b = self.session_id, self.model, body
                rid = self._submit(lambda: {
                    "object": "log_probs_result", "session_id": sid,
                    **model.log_probs(b.get("batch", [])),
                })
                self._respond(200, {"request_id": rid})

            elif path == "/eval":
                sid, model, b = self.session_id, self.model, body
                rid = self._submit(lambda: {
                    "object": "eval_result", "session_id": sid,
                    "step": model.step_count,
                    **model.eval(
                        b.get("data"),
                        b.get("metrics"),
                        b.get("max_examples"),
                        b.get("batch_size", 32),
                    ),
                })
                self._respond(200, {"request_id": rid})

            elif path == "/retrieve":
                # Long-poll: block up to 45 s, return result or try_again.
                rid = body.get("request_id")
                if not rid:
                    self._respond(400, {"error": "request_id required"})
                    return
                result = self._store.wait(rid, timeout=45.0)
                if result is None:
                    self._respond(200, {"type": "try_again", "request_id": rid})
                else:
                    self._respond(200, result)

            elif path == "/save":
                # Synchronous — server needs the result to persist to DB.
                result = self.model.save(
                    body.get("name"),
                    body.get("metadata") or {},
                    self.storage_path,
                    self.session_id,
                )
                self._respond(200, {"object": "checkpoint", **result})

            elif path == "/shutdown":
                self._respond(200, {"status": "shutting_down"})
                self.shutdown_flag.append(True)

            else:
                self._respond(404, {"error": "unknown path"})

        except Exception as e:
            logger.exception("Worker handler error on %s", path)
            self._respond(500, {"error": str(e)})

    def _respond(self, status: int, body: dict):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="TT-Train session worker")
    p.add_argument("--session-id", required=True)
    p.add_argument("--api-url", required=True)
    p.add_argument("--api-key", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--lora-config", default="{}")
    p.add_argument("--optimizer-config", default="{}")
    p.add_argument("--storage-path", default="/tmp/tt_train_storage")
    p.add_argument("--port", type=int, default=0)
    return p.parse_args()


def find_free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def register_with_api(api_url: str, api_key: str, session_id: str, worker_url: str):
    url = f"{api_url.rstrip('/')}/internal/sessions/{session_id}/ready"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    for attempt in range(10):
        try:
            r = httpx.post(url, json={"worker_url": worker_url}, headers=headers, timeout=10.0)
            r.raise_for_status()
            logger.info("Registered with API server as %s", worker_url)
            return
        except Exception as e:
            logger.warning("Registration attempt %d failed: %s", attempt + 1, e)
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError("Could not register with API server after 10 attempts")


def main():
    args = parse_args()
    lora_config = json.loads(args.lora_config)
    optimizer_config = json.loads(args.optimizer_config)

    # Try real model; fall back to simulation
    try:
        import ttml  # noqa: F401
        logger.info("ttml available — loading real model")
        model = RealModelState(
            model_id=args.model,
            optimizer_config=optimizer_config,
            storage_path=args.storage_path,
            session_id=args.session_id,
        )
    except ImportError:
        logger.warning("ttml not available — using simulation (no TT hardware)")
        model = SimModelState(args.model, lora_config, optimizer_config)

    hostname = socket.gethostname()
    port = args.port if args.port else find_free_port()
    worker_url = f"http://{hostname}:{port}"

    WorkerHandler.model = model
    WorkerHandler.session_id = args.session_id
    WorkerHandler.storage_path = args.storage_path
    shutdown_flag: list = []
    WorkerHandler.shutdown_flag = shutdown_flag
    WorkerHandler._store = RequestStore()
    WorkerHandler._executor = ThreadPoolExecutor(max_workers=1)  # serial: one compute op at a time

    register_with_api(args.api_url, args.api_key, args.session_id, worker_url)

    server = ThreadedHTTPServer(("0.0.0.0", port), WorkerHandler)
    logger.info("Session worker %s listening on %s", args.session_id, worker_url)

    t = Thread(target=server.serve_forever, daemon=True)
    t.start()

    try:
        while not shutdown_flag:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass

    logger.info("Session worker %s shutting down", args.session_id)
    server.shutdown()


if __name__ == "__main__":
    main()
