#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Demo: Interactive training session.

Creates one session, runs a manual forward_backward + step loop,
samples completions before and after, saves a checkpoint, closes.

    python demo_session.py
"""

import tt_train as tt

tt.api_key = "demo-key"
tt.base_url = "http://localhost:8000/v1"

TRAIN_BATCH = [
    {"input_ids": [1, 2, 3, 4, 5], "labels": [-100, -100, 3, 4, 5]},
    {"input_ids": [6, 7, 8, 9, 10], "labels": [-100, 7, 8, 9, 10]},
]

PROMPTS = [
    "What is the capital of Germany?",
    "Name a planet in our solar system.",
]


def section(title):
    print(f"\n{'='*55}\n  {title}\n{'='*55}")


# 1. Create session
section("1. Create session")
print("  Provisioning (loading model on worker)...")
session = tt.sessions.create(
    model="tt://catalog/tinyllama",
    lora={"rank": 16, "alpha": 32},
    optimizer={"type": "adamw", "lr": 2e-5},
    idle_timeout_minutes=10,
    name="demo-session",
    wait=True,
    wait_timeout=120.0,
)
print(f"  Ready: {session.id}  model={session.model}")

# 2. Sample before training
section("2. Sample before training")
result = session.sample(PROMPTS, max_tokens=64, temperature=0.7)
for i, choice in enumerate(result.choices):
    text = choice.get("text") or choice.get("content", "")
    print(f"  Q: {PROMPTS[i]}\n  A: {text!r}\n")

# 3. Training loop
section("3. Training loop (10 steps)")
for _ in range(10):
    fb = session.forward_backward(TRAIN_BATCH, loss="cross_entropy")
    sr = session.step()
    print(f"  step {sr.step_number:>3}  loss={fb.loss:.4f}  lr={sr.learning_rate:.2e}")

# 4. Sample after training
section("4. Sample after training")
result = session.sample(PROMPTS, max_tokens=64, temperature=0.7)
for i, choice in enumerate(result.choices):
    text = choice.get("text") or choice.get("content", "")
    print(f"  Q: {PROMPTS[i]}\n  A: {text!r}\n")

# 5. Save and close
section("5. Save checkpoint and close")
ckpt = session.save(name="demo-step10")
print(f"  Checkpoint: {ckpt.model_path}  step={ckpt.step}")
session.close()
print(f"  Session closed.")
