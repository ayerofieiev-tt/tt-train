# Scheduler / Cluster Backend Interface

**Version:** v1
**Status:** Authoritative contract

This document defines the interface between the **Scheduler** (the async background loop inside the Training Service) and the **Cluster Backend** (the pluggable layer that submits and manages workloads on compute infrastructure).

- **Scheduler** — polls the database, decides what to dispatch, drives job and session lifecycle transitions
- **Cluster Backend** — abstracts the compute substrate; current implementations are `LocalBackend` (subprocess) and `SlurmBackend` (HPC batch)

Both sides must comply. A new backend implementation must satisfy the `ClusterBackend` ABC exactly. Changes to method signatures or semantics require updating all implementations and the Scheduler.

---

## Deployment Topology

```
Training Service (FastAPI)
  └─ Scheduler (background asyncio loop)
       └─ ClusterBackend (local | slurm)
            ├─ LocalBackend  — asyncio.create_subprocess_exec()
            └─ SlurmBackend  — sbatch / scancel / squeue
```

The Scheduler is started and stopped in the FastAPI `lifespan` hook. It runs a single async loop; `_tick()` is never called concurrently with itself.

---

## ClusterBackend ABC

```python
BackendState = Literal["pending", "running", "done", "failed", "cancelled"]
```

Every backend must implement the following four methods. All are async.

---

### `submit_job()`

```python
async def submit_job(
    *,
    job_id:    str,
    script_path: str,       # absolute path to workers/job_runner.py
    args:      list[str],   # argv passed verbatim to the script
    nodes:     int = 1,
    partition: str | None = None,
    account:   str | None = None,
    env:       dict[str, str] | None = None,
) -> str                    # opaque backend_job_id
```

Submit a batch training job. Called once per job, when the Scheduler observes `status == "queued"`.

On success the Scheduler immediately updates the job record:

```
status      = "running"
slurm_job_id = <returned backend_job_id>
started_at  = now()
```

On exception the Scheduler sets:

```
status = "failed"
error  = {"type": "scheduler_error", "message": str(e)}
```

**Invariant:** `submit_job()` must return before the worker process has necessarily started — it only guarantees the workload has been handed to the substrate.

---

### `submit_session()`

```python
async def submit_session(
    *,
    session_id:  str,
    script_path: str,       # absolute path to workers/session_worker.py
    args:        list[str],
    nodes:       int = 1,
    partition:   str | None = None,
    account:     str | None = None,
    env:         dict[str, str] | None = None,
) -> str                    # opaque backend_job_id
```

Submit a long-running session worker. Called once per session, when the Scheduler observes `status == "provisioning"`.

On success the Scheduler stores the backend ID:

```
slurm_job_id = <returned backend_job_id>
```

Status remains `"provisioning"` — the transition to `"ready"` happens later, when the session worker calls `POST /internal/sessions/{id}/ready`. See [Worker/Server Interface](./worker-server-interface.md).

On exception the Scheduler sets `status = "failed"`.

---

### `cancel()`

```python
async def cancel(backend_job_id: str) -> None
```

Terminate a running workload. Best-effort, non-blocking. Implementations must not raise for workloads that have already exited. Called on explicit user cancellation and on scheduler shutdown.

---

### `get_state()`

```python
async def get_state(backend_job_id: str) -> BackendState
```

Return the current state of a workload. Safe for concurrent calls. Returns a snapshot; callers must not assume the state is still current by the time they act on it.

| `BackendState` | Meaning |
|---|---|
| `pending` | Submitted but not yet executing |
| `running` | Currently executing |
| `done` | Exited with code 0 |
| `failed` | Exited with non-zero code or unrecoverable error |
| `cancelled` | Killed by signal or explicit cancel |

> **Note:** The Scheduler does not currently poll `get_state()` in its main tick. The method exists for monitoring integrations and future reconciliation logic. State transitions in the DB are driven by worker callbacks, not by backend polling.

---

### Concurrency caps

Each backend exposes two optional integer limits:

```python
max_concurrent_jobs:     int | None   # None = no cap (backend owns scheduling)
max_concurrent_sessions: int | None
```

When set, the Scheduler counts currently `running` jobs / `ready + provisioning` sessions before submitting new ones, and submits only enough to stay within the cap. When `None`, the Scheduler submits up to 10 queued items per tick and lets the backend manage concurrency.

---

## Scheduler Tick

The Scheduler calls `_tick()` every `scheduler_poll_interval` seconds (default: 5.0 s). One tick is:

1. **Dispatch queued jobs** — fetch `status="queued"` jobs, submit each via `submit_job()`, update DB
2. **Dispatch provisioning sessions** — fetch `status="provisioning"` sessions, submit via `submit_session()`, update DB
3. **Expire idle sessions** — for each `status="ready"` session where `(now − last_active_at) > idle_timeout_minutes`, POST to `{worker_url}/shutdown` (best-effort, 10 s timeout), then set `status="expired"`

Tick exceptions are caught and logged; the loop continues on the next interval.

---

## Script Launch Protocol

The Scheduler constructs `args` and passes them to the backend. The backend must forward them verbatim as argv to the script.

### Job runner args

```
--job-id         <job.id>
--api-url        <settings.api_base_url>
--api-key        <settings.internal_api_key>
--model          <job.model>
--method         <job.method>
--training-data  <job.training_data>          (omitted if None)
--validation-data <job.validation_data>       (omitted if None)
--config         <json(job.config)>
--storage-path   <settings.shared_storage_path>
--console-job-id <job.console_job_id>         (omitted if None)
--dataset-url    <job.dataset_url>            (omitted if None)
--console-base-url <job.console_base_url>     (omitted if None)
--worker-token   <job.worker_token>           (omitted if None)
--callback-url   <job.callback_url>           (omitted if None)
```

### Session worker args

```
--session-id       <session.id>
--api-url          <settings.api_base_url>
--api-key          <settings.internal_api_key>
--model            <session.model>
--lora-config      <json(session.lora_config)>
--optimizer-config <json(session.optimizer_config)>
--storage-path     <settings.shared_storage_path>
```

### PYTHONPATH injection

Both backends inject the repo root into `PYTHONPATH` so workers can import `from workers.common import ...`. Backends must merge with the existing environment rather than replacing it.

---

## LocalBackend

Spawns workers as subprocesses via `asyncio.create_subprocess_exec()`.

| Property | Value |
|---|---|
| Backend job ID (jobs) | `local-{n}` |
| Backend job ID (sessions) | `local-sess-{n}` |
| Log output | `/tmp/tt_train_logs/{job,sess}_{id}.{out,err}` |
| Default concurrency cap (jobs) | 1 |
| Default concurrency cap (sessions) | 1 |

**Cancellation:** SIGTERM → wait 5 s → SIGKILL. Silent if process already exited.

**State query:** inspects `/proc/{pid}/status` to detect silent death. Returns `"done"` for any unknown or already-removed process handle.

---

## SlurmBackend

Submits workloads via `sbatch` and queries state via `squeue`.

| Property | Value |
|---|---|
| Backend job ID | Slurm numeric job ID (e.g. `"12345"`) |
| Log output | `/tmp/tt_train_logs/{job,sess}_{id}.{out,err}` |
| Default concurrency cap | `None` (Slurm owns scheduling) |

**sbatch script template:**

```bash
#!/bin/bash
#SBATCH --job-name=tt-{job|sess}-{id}
#SBATCH --nodes={nodes}
#SBATCH --output={stdout_path}
#SBATCH --error={stderr_path}
[#SBATCH --partition={partition}]   # if settings.slurm_partition is set
[#SBATCH --account={account}]       # if settings.slurm_account is set

[source {slurm_venv_path}/bin/activate]   # if settings.slurm_venv_path is set
export PYTHONPATH="{repo_root}:${PYTHONPATH}"
python "{script_path}" {args...}
```

Script is written to `settings.slurm_script_tmpdir` (default: `/tmp/tt_train_sbatch`), made executable, submitted, then deleted.

**Slurm state mapping:**

| Slurm state(s) | `BackendState` |
|---|---|
| `PENDING`, `CONFIGURING`, `RESIZING`, `SUSPENDED` | `pending` |
| `RUNNING`, `COMPLETING` | `running` |
| `COMPLETED` | `done` |
| `FAILED`, `TIMEOUT`, `NODE_FAIL`, `PREEMPTED`, `BOOT_FAIL`, `DEADLINE`, `OUT_OF_MEMORY` | `failed` |
| `CANCELLED`, `CANCELLED+`, `REVOKED` | `cancelled` |

Non-zero `squeue` exit (job has left the queue) is treated as `"done"`.

**Cancellation:** `scancel {backend_job_id}`. "Invalid job id" errors are suppressed (job already finished). Other `scancel` errors raise `RuntimeError`.

---

## Configuration

All settings are prefixed `TT_TRAIN_` or read from `.env`.

| Variable | Default | Description |
|---|---|---|
| `CLUSTER_BACKEND` | `local` | `local` or `slurm` |
| `SCHEDULER_POLL_INTERVAL` | `5.0` | Seconds between scheduler ticks |
| `SESSION_IDLE_TIMEOUT_MINUTES` | `30` | Global default idle timeout for sessions |
| `LOCAL_MAX_CONCURRENT_JOBS` | `1` | LocalBackend job concurrency cap |
| `LOCAL_MAX_CONCURRENT_SESSIONS` | `1` | LocalBackend session concurrency cap |
| `WORKER_SCRIPT_DIR` | `/home/boxx/tt-train/workers` | Directory containing worker scripts |
| `SHARED_STORAGE_PATH` | `/tmp/tt_train_storage` | Shared filesystem root |
| `API_BASE_URL` | `http://localhost:8000/v1` | URL workers call back to |
| `INTERNAL_API_KEY` | `internal-secret` | Bearer token workers use for callbacks |
| `SLURM_PARTITION` | None | Slurm partition (passed to sbatch) |
| `SLURM_ACCOUNT` | None | Slurm account (passed to sbatch) |
| `SLURM_VENV_PATH` | `""` | Venv to activate on compute nodes |
| `SLURM_SCRIPT_TMPDIR` | `/tmp/tt_train_sbatch` | Temp directory for sbatch scripts |

---

## Database State Transitions

### Jobs

```
queued
  │  Scheduler.submit_job() → backend.submit_job()
  ▼
running  ←─ slurm_job_id = backend_id, started_at = now()
  │
  ├─► completed   (worker calls /internal/jobs/{id}/complete)
  ├─► failed      (worker calls /internal/jobs/{id}/fail, or submission error)
  └─► cancelled   (worker receives SIGTERM → reports fail with type="cancelled")
```

### Sessions

```
provisioning
  │  Scheduler.submit_session() → backend.submit_session()
  │  (slurm_job_id = backend_id; status unchanged)
  ▼
provisioning  ←─ waiting for worker to call /internal/sessions/{id}/ready
  │
  ▼
ready  ←─ worker_url = "http://{host}:{port}", last_active_at = now()
  │
  ├─► closed    (user calls DELETE /sessions/{id})
  ├─► expired   (scheduler idle-timeout check: now - last_active_at > idle_timeout)
  └─► failed    (submission error)
```

---

## Error Handling

| Failure | Scheduler behaviour |
|---|---|
| `submit_job()` raises | Job set to `failed`; error recorded; scheduler continues |
| `submit_session()` raises | Session set to `failed`; scheduler continues |
| `cancel()` raises | Logged; not propagated; scheduler continues |
| Tick raises unexpectedly | Logged; loop resumes on next interval |
| Worker dies silently (LocalBackend) | Detected via `/proc/{pid}` check on next `get_state()` call |

The Scheduler does not apply exponential backoff to submission retries. A failed submission stays in the DB as `failed`. To retry, the user must create a new job or session.

---

## Responsibility Boundary

### Scheduler owns

- Polling the database for work
- Enforcing concurrency caps (LocalBackend mode)
- Calling `submit_job()` / `submit_session()` and updating `slurm_job_id` + `started_at`
- Idle session detection and expiry
- Setting terminal `failed` status on submission errors

### ClusterBackend owns

- Launching the worker process on the correct substrate
- Generating a stable, opaque `backend_job_id` for tracking
- PYTHONPATH and environment setup for worker processes
- Cancellation (SIGTERM/SIGKILL or scancel)
- State reporting via `get_state()`

### Workers own

- All subsequent status transitions (running → completed/failed) via internal callbacks
- Progress and metrics reporting
- Session readiness registration
