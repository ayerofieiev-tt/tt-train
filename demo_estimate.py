#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Demo 3: Explore the platform — models, hardware, cost estimation, datasets.

A read-heavy demo that shows how to inspect what's available before
committing to a training run. Good for onboarding or a product walkthrough.

Run the server first:
    uvicorn server.main:app --port 8000

Then:
    python demo_estimate.py
"""

import io
import json

import tt_train as tt

tt.api_key = "demo-key"
tt.base_url = "http://localhost:8000/v1"


def section(title: str) -> None:
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


# ---------------------------------------------------------------------------
# 1. Browse the model catalog
# ---------------------------------------------------------------------------
section("1. Model catalog")

models = tt.models.list()
for m in models.data:
    status = getattr(m, "status", "available")
    ctx = getattr(m, "context_length", "?")
    print(f"  {m.id:<40}  ctx={ctx}  status={status}")


# ---------------------------------------------------------------------------
# 2. Check hardware availability
# ---------------------------------------------------------------------------
section("2. Hardware availability")

hw = tt.hardware.list()
for card in hw.data:
    available = getattr(card, "available", "?")
    name = getattr(card, "name", card.id)
    print(f"  {name:<30}  available={available}")


# ---------------------------------------------------------------------------
# 3. Cost estimate — compare methods and model sizes
# ---------------------------------------------------------------------------
section("3. Cost estimates")

configs = [
    ("tt://catalog/tinyllama",    "sft", {"max_steps": 1000, "lr": 2e-5}),
    ("tt://catalog/llama-3.2-1b", "sft", {"max_steps": 1000, "lr": 2e-5}),
    ("tt://catalog/llama-3.2-1b", "dpo", {"max_steps": 500,  "beta": 0.1}),
]

for model, method, config in configs:
    est = tt.jobs.estimate(model, method, training_data="ds_placeholder", config=config)
    print(
        f"  {model.split('/')[-1]:<20}  method={method:<5}  "
        f"cost≈{est.estimated_cost}  time≈{est.estimated_time_seconds//60}min  "
        f"steps≈{est.estimated_steps}"
    )


# ---------------------------------------------------------------------------
# 4. Upload a dataset, inspect it, then delete it
# ---------------------------------------------------------------------------
section("4. Dataset lifecycle")

EXAMPLES = [
    {"messages": [
        {"role": "user", "content": "Explain gradient descent in one sentence."},
        {"role": "assistant", "content": "Gradient descent updates model parameters by stepping in the direction that reduces loss."},
    ]},
    {"messages": [
        {"role": "user", "content": "What is a transformer?"},
        {"role": "assistant", "content": "A transformer is a neural network architecture based on self-attention mechanisms."},
    ]},
]

jsonl = io.BytesIO("\n".join(json.dumps(ex) for ex in EXAMPLES).encode())
ds = tt.datasets.create(jsonl, format="conversational", name="explore-demo", description="Temporary demo dataset")

print(f"  Uploaded:  {ds.id}  examples={ds.stats.examples if ds.stats else '?'}")

# List all datasets
page = tt.datasets.list()
print(f"  Total datasets on server: {len(page.data)}")
for d in page.data:
    examples = d.stats.examples if d.stats else "?"
    print(f"    {d.id}  {d.name or '(unnamed)'}  examples={examples}")

# Delete the one we just created
tt.datasets.delete(ds.id)
print(f"  Deleted: {ds.id}")

page2 = tt.datasets.list()
print(f"  Datasets remaining: {len(page2.data)}")


# ---------------------------------------------------------------------------
# 5. List any existing jobs from previous demos
# ---------------------------------------------------------------------------
section("5. Existing jobs (from previous demos)")

jobs = tt.jobs.list(limit=5)
if not jobs.data:
    print("  No jobs yet. Run demo_jobs.py first.")
else:
    for j in jobs.data:
        loss = (j.metrics or {}).get("train_loss", "-") if j.metrics else "-"
        print(f"  {j.id}  {j.status:<12}  method={j.method}  loss={loss}  name={j.name}")
