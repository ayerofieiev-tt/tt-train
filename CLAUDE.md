# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

`tt-train` is a Python SDK for the TT-Train API — fine-tuning and training LLMs on Tenstorrent hardware. It provides two primary programming models:

1. **Jobs API** — black-box fine-tuning: upload data, launch a job, stream progress, get a result model
2. **Sessions API** — interactive training with low-level primitives: `forward_backward()`, `step()`, `sample()`, `eval()`, `save()`

## Commands

Install dev dependencies:
```bash
pip install -e ".[dev]"
```

Run all tests:
```bash
pytest
```

Run a single test:
```bash
pytest tests/test_sdk.py::test_create_job
```

Lint:
```bash
ruff check .
```

Format:
```bash
ruff format .
```

## Architecture

```
tt_train/
  __init__.py       # Module-level API (tt.jobs.create(...)) via _ModuleProxy
  client.py         # Client class — assembles all resource objects
  http.py           # HTTPClient — auth, retries, SSE streaming, file upload
  types.py          # All Pydantic models (Job, Session, Dataset, etc.)
  errors.py         # Exception hierarchy + raise_for_error() dispatcher
  resources/
    jobs.py         # Jobs resource (create, get, list, cancel, stream, wait, estimate)
    sessions.py     # Sessions + Session class (the interactive training handle)
    datasets.py     # Dataset upload and management
    models.py       # Model catalog and checkpoint management
    inference.py    # Inference on trained models
    rewards.py      # Remote reward functions for RL
    hardware.py     # Hardware catalog and availability
```

### Key Design Patterns

**Dual API surface:** Users can use module-level globals (`tt.jobs.create(...)`) or an explicit `Client` instance. The module-level globals are `_ModuleProxy` objects that lazily create a default client from `tt.api_key` / `TT_TRAIN_API_KEY` env var on first access.

**HTTPClient** (`http.py`) is the single point for all network I/O. It handles:
- Bearer token auth + optional org/project headers
- Automatic retries with exponential backoff for 429/5xx
- SSE streaming via `stream_sse()` (used by `jobs.stream()`)
- Multipart file upload via `upload()`

**Session vs Sessions:** `Sessions` is the resource manager (create/get/list). `Session` is the stateful handle returned by `Sessions.create()` — it tracks `step_count` locally and exposes all training primitives. `Session` supports context manager usage.

**Types:** All API responses are validated through Pydantic models in `types.py`. Resources call `SomeModel.model_validate(data)` on the raw dict from `HTTPClient`.

### Testing

Tests use `respx` to mock `httpx` at the transport level. The `client` fixture creates a `Client(api_key="tt-test-key")`. Tests do not hit any real network endpoints.
