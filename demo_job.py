#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Demo: Black-box SFT job.

Uploads a dataset, creates one job, streams progress, prints result.

    python demo_job.py
"""

import io
import json

import tt_train as tt

tt.api_key = "demo-key"
tt.base_url = "http://localhost:8000/v1"

EXAMPLES = [
    {"messages": [
        {"role": "user", "content": "What is 2 + 2?"},
        {"role": "assistant", "content": "4"},
    ]},
    {"messages": [
        {"role": "user", "content": "Name a primary colour."},
        {"role": "assistant", "content": "Red"},
    ]},
    {"messages": [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "Paris"},
    ]},
] * 10


def section(title):
    print(f"\n{'='*55}\n  {title}\n{'='*55}")


# 1. Upload dataset
section("1. Upload dataset")
jsonl = io.BytesIO("\n".join(json.dumps(ex) for ex in EXAMPLES).encode())
dataset = tt.datasets.create(jsonl, format="conversational", name="demo-qa")
examples = dataset.stats.examples if dataset.stats else "?"
print(f"  {dataset.id}  examples={examples}")

# 2. Create job
section("2. Create SFT job")
job = tt.jobs.create(
    model="tt://catalog/tinyllama",
    method="sft",
    training_data=dataset.id,
    config={"max_steps": 20, "lr": 2e-5},
    name="demo-job",
)
print(f"  {job.id}  status={job.status}")

# 3. Stream progress
section(f"3. Streaming {job.id}")
print("  (Ctrl-C to skip)\n")

try:
    for event in tt.jobs.stream(job.id):
        if event.event == "job.progress":
            d = event.data
            step = d.get("step", 0)
            total = d.get("total_steps", 1) or 1
            pct = d.get("percentage", step / total * 100)
            loss = d.get("loss", 0.0)
            bar = "#" * int(pct / 5)
            print(f"\r  [{bar:<20}] {pct:5.1f}%  step {step}/{total}  loss={loss:.4f}",
                  end="", flush=True)
        elif event.event == "job.completed":
            print(f"\n  Completed!  result_model={event.data.get('result_model')}")
        elif event.event == "job.failed":
            print(f"\n  Failed: {event.data.get('error')}")
        elif event.event == "done":
            break
except KeyboardInterrupt:
    print("\n  (skipped — waiting for completion)")
    job = tt.jobs.wait(job.id, poll_interval=3.0)

job = tt.jobs.get(job.id)
loss = (job.metrics or {}).get("train_loss", "-")
print(f"\n  status={job.status}  loss={loss}")
