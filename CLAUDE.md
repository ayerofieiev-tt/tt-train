# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

`tt-train` is a Python SDK for the TT-Train API — fine-tuning and training LLMs on Tenstorrent hardware. It provides two primary programming models:

1. **Jobs API** — black-box fine-tuning: upload data, launch a job, stream progress, get a result model
2. **Sessions API** — interactive training with low-level primitives: `forward_backward()`, `step()`, `sample()`, `eval()`, `save()`

The repo contains three components: the **SDK** (`tt_train/`), the **API server** (`server/`), and the **worker processes** (`workers/`).

## Commands

Install dev dependencies:
```bash
pip install -e ".[dev]"
```

Install server dependencies:
```bash
pip install -e ".[server]"
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

Run the server (defaults to SQLite, local cluster backend):
```bash
uvicorn server.main:app --reload --port 8000
```

## SDK Architecture (`tt_train/`)

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

**Dual API surface:** Module-level globals (`tt.jobs.create(...)`) or an explicit `Client` instance. The globals are `_ModuleProxy` objects that lazily create a default client from `tt.api_key` / `TT_TRAIN_API_KEY` on first access.

**HTTPClient** is the single point for all network I/O: Bearer token auth, exponential backoff retries for 429/5xx, SSE streaming via `stream_sse()`, and multipart file upload via `upload()`.

**Session vs Sessions:** `Sessions` is the resource manager (create/get/list). `Session` is the stateful handle returned by `Sessions.create()` — tracks `step_count` locally, exposes all training primitives, supports context manager usage.

**APIFuture:** Training primitives (`forward_backward()`, `step()`, `sample()`, `eval()`, `log_probs()`) return an `APIFuture[T]` rather than a result directly. The server accepts the command immediately and returns a `request_id`; the future's `.result()` long-polls `/sessions/{id}/retrieve` (server holds for 45s, client timeout is 55s) until the computation completes. Call `.result(timeout=N)` for an overall deadline, or `await future` in async code.

**Types:** All API responses are validated through Pydantic models in `types.py`. Resources call `SomeModel.model_validate(data)` on the raw dict from `HTTPClient`.

**Testing:** Tests use `respx` to mock `httpx` at the transport level. The `client` fixture creates a `Client(api_key="tt-test-key")`. Tests do not hit any real network endpoints.

## Server Architecture (`server/`)

The server is a FastAPI app that persists state to a database and dispatches work to a cluster backend.

```
server/
  main.py           # FastAPI app, startup/shutdown hooks, router mounts
  config.py         # Pydantic Settings — all config via TT_TRAIN_* env vars or .env
  auth.py           # Bearer token auth dependency (verify_auth)
  store.py          # In-memory store (model catalog, hardware catalog, model checkpoints) + new_id()/now_iso() utils
  db/
    engine.py       # SQLAlchemy async engine + Base + get_db() dependency
    models.py       # ORM models: Job, Session, Dataset, Checkpoint, RewardFunction
    crud.py         # Async CRUD helpers used by routers and scheduler
  routers/
    jobs.py         # Jobs CRUD + SSE stream + estimate
    sessions.py     # Sessions CRUD + training primitive endpoints
    datasets.py     # Multipart dataset upload + CRUD
    models.py       # Model catalog + checkpoints + download URLs
    inference.py    # Generate (streaming SSE + non-streaming) + batch + chat/completions
    rewards.py      # Reward function upload + test
    hardware.py     # Hardware catalog
    internal.py     # Worker callbacks (/internal/jobs/{id}/progress|complete|fail, /internal/sessions/*)
  cluster/
    base.py         # ClusterBackend ABC + BackendState type
    local.py        # LocalBackend — spawns workers as subprocesses, enforces concurrency cap
    slurm.py        # SlurmBackend — submits via sbatch/scancel/squeue
  scheduler/
    service.py      # Scheduler — async background loop: dispatches queued jobs/sessions, expires idle sessions
```

**State split:** Jobs, sessions, datasets, and reward functions are persisted in the DB (SQLAlchemy). The model catalog and hardware catalog live in the in-memory `store.py` (static data pre-populated at import time). All routers import `new_id()` / `now_iso()` from `store.py`.

**Request flow:** Client → router (auth, validate, CRUD) → DB. Scheduler polls DB every `scheduler_poll_interval` seconds, submits queued items to the cluster backend, and marks idle sessions as expired.

**Internal callbacks:** Workers call back to `/v1/internal/*` using `TT_TRAIN_INTERNAL_API_KEY` to report progress, completion, or failure. The internal router updates the DB accordingly.

**Database:** Defaults to SQLite (`sqlite+aiosqlite:///./tt_train.db`). Set `TT_TRAIN_DATABASE_URL` to a PostgreSQL URL for production. Tables are auto-created on startup (no Alembic migrations yet).

**Key config env vars** (prefix `TT_TRAIN_`, or set in `.env`):

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | SQLite file | SQLAlchemy async DB URL |
| `CLUSTER_BACKEND` | `local` | `local` or `slurm` |
| `INTERNAL_API_KEY` | `internal-secret` | Workers use this to call back |
| `API_BASE_URL` | `http://localhost:8000/v1` | How workers reach the server |
| `WORKER_SCRIPT_DIR` | `/home/boxx/tt-train/workers` | Where worker scripts live |
| `SHARED_STORAGE_PATH` | `/tmp/tt_train_storage` | Shared filesystem for datasets/checkpoints |
| `SLURM_PARTITION` / `SLURM_ACCOUNT` | None | Slurm-specific |

## Workers (`workers/`)

Workers are standalone scripts launched by the scheduler as subprocesses (local) or Slurm batch jobs.

- `job_runner.py` — Runs SFT training. Attempts real training via `ttml`/`ttnn` on TT hardware; falls back to a simulation loop if the TT stack is unavailable. Reports progress to `/v1/internal/jobs/{id}/progress` via `JobReporter`. Uses `TelemetrySFTTrainer` (dynamic subclass of `ttml.SFTTrainer`) to hook into checkpointing, plus a `ProgressPoller` thread that reports every 30s between checkpoints.
- `session_worker.py` — Hosts an interactive training session, exposing an HTTP server that the API server proxies training primitives through.
- `common.py` — Shared constants: `MODEL_CATALOG` (maps `tt://catalog/...` IDs to HuggingFace repos and ttml config paths), `TTML_CONFIG_DIR`.
- `events.py` — `EventEmitter` protocol + `HttpEventEmitter` / `NoopEventEmitter`. Workers call `emit()` to post usage/billing events to a platform metering endpoint (`/v1/events`). Activated when `TT_TRAIN_PLATFORM_BASE_URL` and `TT_TRAIN_WORKER_TOKEN` are set; otherwise falls back to `NoopEventEmitter`.

Datasets are expected at `{shared_storage}/datasets/{dataset_id}/data.jsonl` (JSONL with `messages` field). Checkpoints are written to `{shared_storage}/checkpoints/{job_id}/`.
