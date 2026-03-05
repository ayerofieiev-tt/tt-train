# tt-train

Python SDK for the TT-Train API — fine-tuning and training on Tenstorrent hardware.

## Install

```bash
pip install tt-train
```

## Quick Start

### Black-box fine-tuning (Jobs API)

```python
import tt_train as tt

tt.api_key = "tt-..."

# Upload training data
ds = tt.datasets.create("train.jsonl", format="chat", name="my-data")
ds = tt.datasets.wait_until_ready(ds.id)
print(f"Dataset: {ds.stats.examples} examples, {ds.stats.tokens} tokens")

# Estimate cost
estimate = tt.jobs.estimate(
    model="tt://catalog/llama-3.2-8b",
    method="sft",
    training_data=ds.id,
    config={"epochs": 3, "lora": {"rank": 64}},
)
print(f"Estimated: {estimate.estimated_cost} in {estimate.estimated_time_seconds}s")

# Launch training
job = tt.jobs.create(
    model="tt://catalog/llama-3.2-8b",
    method="sft",
    training_data=ds.id,
    config={
        "epochs": 3,
        "lr": 2e-5,
        "lora": {"rank": 64, "alpha": 128},
    },
    name="my-first-finetune",
)

# Stream progress
for event in tt.jobs.stream(job.id):
    if event.event == "metrics":
        print(f"step {event.data['step']}: loss={event.data['train_loss']:.4f}")
    elif event.event == "completed":
        print(f"Done! Model: {event.data['result_model']}")
        break

# Or just wait
job = tt.jobs.wait(job.id)
print(f"Result model: {job.result_model}")

# Test inference
response = tt.inference.generate(
    model=job.result_model,
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message["content"])
```

### Interactive training (Sessions API)

```python
import tt_train as tt

tt.api_key = "tt-..."

# Create a session — blocks until hardware is ready
session = tt.sessions.create(
    model="tt://catalog/llama-3.2-8b",
    lora={"rank": 64, "alpha": 128},
    optimizer={"type": "adamw", "lr": 2e-5},
)

# SFT training loop
for epoch in range(3):
    for batch in my_dataloader:
        result = session.forward_backward(batch=batch, loss="cross_entropy")
        session.step()
        print(f"step {session.step_count}: loss={result.loss:.4f}, cost={result.cost}")

    # Evaluate
    metrics = session.eval("ds_val_123", metrics=["loss", "perplexity"])
    print(f"epoch {epoch}: val_loss={metrics.metrics['loss']:.4f}")

    # Checkpoint
    session.save(name=f"epoch-{epoch}")

session.close()
```

### Online RL

```python
session = tt.sessions.create(
    model="tt://catalog/qwen3-8b",
    lora={"rank": 64},
    optimizer={"type": "adamw", "lr": 1e-5},
)

for step in range(1000):
    # Sample from current policy
    completions = session.sample(
        prompts=prompt_batch,
        temperature=0.8,
        n=4,
        max_tokens=512,
    )

    # Score with your reward function (runs locally)
    rewards = compute_rewards(prompt_batch, completions)

    # Policy gradient update
    rl_batch = build_rl_batch(prompt_batch, completions, rewards)
    result = session.forward_backward(batch=rl_batch, loss="grpo", loss_config={"beta": 0.04})
    session.step()

    if step % 100 == 0:
        session.save(name=f"rl-step-{step}")

session.close()
```

### Agent-driven training

```python
session = tt.sessions.create(
    model="tt://catalog/llama-3.2-8b",
    lora={"rank": 64},
    optimizer={"type": "adamw", "lr": 2e-5},
)

# Train initial round
for batch in initial_data:
    session.forward_backward(batch=batch, loss="cross_entropy")
    session.step()

# Evaluate and decide
metrics = session.eval(eval_data, metrics=["loss", "accuracy"])

while metrics.metrics["accuracy"] < 0.95:
    # Agent identifies weak spots and gathers more data
    failures = find_failures(session, eval_data)
    targeted_data = generate_targeted_examples(failures)

    for batch in targeted_data:
        session.forward_backward(batch=batch, loss="cross_entropy")
        session.step()

    metrics = session.eval(eval_data, metrics=["loss", "accuracy"])
    print(f"accuracy: {metrics.metrics['accuracy']:.3f}, cost so far: {session.total_cost}")

ckpt = session.save(name="agent-approved")
print(f"Final model: {ckpt.model_path}")
session.close()
```

### Explicit client (recommended for libraries/services)

```python
from tt_train import Client

with Client(api_key="tt-...", organization="org_abc") as client:
    job = client.jobs.create(
        model="tt://catalog/llama-3.2-8b",
        method="sft",
        training_data="ds_123",
    )
    job = client.jobs.wait(job.id)
```

## Configuration

| Setting | Env Variable | Default |
|---------|-------------|---------|
| API Key | `TT_TRAIN_API_KEY` | — |
| Base URL | `TT_TRAIN_BASE_URL` | `https://api.tt-train.dev/v1` |

## Hardware

TT-Train runs on Tenstorrent accelerators:

| Accelerator | Memory | Best For |
|------------|--------|----------|
| Wormhole | 12GB/device | Models ≤ 13B (LoRA) |
| Blackhole | 32GB/device | Models ≤ 70B (LoRA), MoE |

Check availability and pricing:

```python
catalog = tt.hardware.catalog()
for acc in catalog.accelerators:
    print(f"{acc.name}: {acc.available_nodes} nodes, {acc.pricing['standard'].per_node_hour}/node-hr")
```

## License

Apache 2.0
