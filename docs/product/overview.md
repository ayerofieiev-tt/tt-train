# Product Overview

## What TT-Train Is

TT-Train is a platform for fine-tuning and training large language models on Tenstorrent hardware. It exposes Tenstorrent accelerators (Wormhole, Blackhole) through a cloud-style API, making specialized AI hardware accessible to ML engineers who want more than a hosted model inference endpoint but less than managing raw hardware directly.

## Problem Being Solved

Training and fine-tuning LLMs requires:
1. Access to purpose-built AI accelerators
2. Non-trivial infrastructure (cluster management, checkpointing, progress tracking)
3. Programming against low-level frameworks (ttml/ttnn) for Tenstorrent hardware

TT-Train abstracts all of this. Users interact with a clean Python SDK — they upload data, describe what they want, and get a trained model back. Alternatively, they can drop down to interactive primitives for research workflows.

## Two Programming Models

### Jobs API — Black-Box Fine-Tuning
For users who want to run a fine-tuning run as a managed operation:
- Upload a dataset, specify a base model and method, launch a job
- Stream real-time progress (loss curves, steps, ETA)
- Retrieve the result model checkpoint when done
- Use cases: production fine-tuning pipelines, scheduled training runs, automated data-flywheel systems

### Sessions API — Interactive Training
For users who want direct control over the training loop:
- Provision a session (allocates hardware, loads model into accelerator memory)
- Call `forward_backward()`, `step()`, `sample()`, `eval()`, `save()` as primitives
- Session stays alive across calls (persistent GPU state), auto-expires after idle
- Use cases: RL fine-tuning with custom reward logic, online learning, research experiments, custom training loop logic

## Target Users

| User | Primary API | Typical Workflow |
|---|---|---|
| **AI agent / autonomous system** | Jobs + Sessions | Agent collects data, uploads it, launches fine-tuning, evaluates, iterates — all programmatically in a closed loop |
| ML engineer running production fine-tunes | Jobs | CI/CD pipeline → `jobs.create()` → stream progress → deploy result |
| Researcher experimenting with RL | Sessions | Interactive loop: generate → score → `forward_backward()` → `step()` |
| Researcher ablating hyperparameters | Jobs + Estimate | `jobs.estimate()` → iterate config → `jobs.create()` → `jobs.wait()` |
| Application developer | Inference | `inference.generate()` against a fine-tuned checkpoint |

### Agents as First-Class Users

The Python SDK is intentionally designed to be called by AI agents, not just humans. An agent can:
- **Self-improve:** Run inference, collect feedback, upload training data, fine-tune, swap in the new checkpoint — fully automated
- **Explore cost/time tradeoffs:** Use `jobs.estimate()` before committing to a run
- **Run RLHF/RLAIF pipelines:** Use the Sessions API to control the training loop step-by-step, scoring completions with custom reward functions and deciding when to checkpoint or stop
- **Online learning:** Keep a session alive, serve requests, fine-tune on live signal with `forward_backward()` + `step()`

The SDK's synchronous, typed API surface (explicit return types, structured error hierarchy, no hidden side effects) makes it easy to call from agent frameworks and tool-use systems.

## Supported Training Methods

| Method | Description |
|---|---|
| `sft` | Supervised fine-tuning (instruction-following, chat) |
| `dpo` | Direct Preference Optimization (alignment from human preferences) |
| `rl` | Reinforcement Learning with remote reward functions |
| `pretrain` | Continued pre-training on domain-specific text |

## Key User-Facing Concepts

**Dataset** — JSONL file in conversational format (`{"messages": [...]}`), uploaded and stored by the platform. Referenced by ID in jobs and sessions.

**Job** — An async training operation. Has a lifecycle: `queued → running → completed | failed | cancelled`. Progress is streamable via SSE. Produces a result model checkpoint.

**Session** — A stateful, persistent training context backed by live hardware. Accepts training commands in real time. Expires after inactivity. Can be checkpointed manually at any step.

**Checkpoint** — A saved model state. Addressable as `tt://checkpoints/{id}/...`. Can be used as input to subsequent jobs or sessions, or downloaded.

**Reward Function** — A Python script uploaded by the user, executed remotely to score model completions during RL fine-tuning.

## Non-Goals (Current Scope)

- Multi-modal training (vision, audio)
- Inference serving at scale (the inference endpoint is for testing trained models, not production serving)
- Automatic hyperparameter search
- Model merging / model ensembling
- Non-Tenstorrent hardware
