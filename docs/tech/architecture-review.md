# Architecture Review: TT-Train ↔ TT-Cloud-Console

**Status:** Draft for discussion
**Scope:** Where responsibilities should live across the two repos, interface alignment, and tech stack decisions

---

## Current State: What Each System Owns

### TT-Cloud-Console (`tt-cloud-console`, Go/Next.js)
- GitHub OAuth, sessions, org management, team RBAC
- API token issuance (`sk-tt-...` tokens, bcrypt-hashed in `api_tokens`)
- Credit billing: Stripe top-ups, `orgs.credit_balance_cents`, transaction ledger, metering
- User-facing dashboard and admin UI (Next.js)
- Inference proxy (`tt-dashboard-proxy`)
- Fine-tuning job tracking for billing: `fine_tuning_jobs`, pre-auth, settlement
- Training data/metrics/checkpoint storage: `training_metrics`, `training_logs`, `training_checkpoints`, `datasets` (S3-backed)
- SSE real-time updates to the browser
- Temporal for long-running workflows
- PostgreSQL (shared across all the above, with RLS)

### TT-Train (`tt-train`, Python/FastAPI)
- Python SDK (`tt_train` package — what users and agents install)
- HTTP API the SDK talks to
- Job scheduling (asyncio loop + LocalBackend / SlurmBackend)
- Training execution (ttml/ttnn integration, `job_runner.py`)
- Interactive Sessions API (`session_worker.py` with forward_backward/step/sample/eval/save)
- Model catalog (`tt://catalog/...`)
- Its own database (SQLite by default, isolated from console's PostgreSQL)
- Dataset management on local filesystem
- Checkpoint storage on local filesystem

---

## The Problems

### 1. Two Job Tables, No Bridge
The console has `fine_tuning_jobs` tracking the billing lifecycle. TT-Train has its own jobs table tracking the execution lifecycle. These are separate UUIDs in separate databases with no FK relationship. Right now `TTTrainService` in the console forwards the request and reads back TT-Train's ID, but the two records have no shared key and the state machines are independently managed.

### 2. Workers Calling the Wrong Place
TT-Train workers call back to TT-Train's `/internal/jobs/{id}/progress|complete|fail`. The console has designed a different callback pattern: job-scoped JWT-authenticated `POST /v1/fine-tuning/jobs/{id}/metrics|logs|checkpoints`. These are completely separate. The metrics/logs that workers produce are currently only in TT-Train's DB — not visible to the console, not stored in `training_metrics` / `training_logs`.

### 3. Auth Is Not Wired
The console issues `sk-tt-...` API tokens (bcrypt-hashed in `api_tokens`). TT-Train accepts any non-empty Bearer token. Production requires TT-Train to validate tokens against the console's `api_tokens` table. There is no validation endpoint or shared DB connection.

### 4. Two Dataset Systems
The console has a `datasets` table + S3 storage with upload/validation flow. TT-Train stores datasets as JSONL files on local filesystem. When the console submits a job to TT-Train it sends `dataset_url` (a URL string), but TT-Train workers expect files at a local path (`{storage_path}/datasets/{id}/data.jsonl`). These don't connect.

### 5. Billing Is Not Triggered
TT-Train workers don't call the `EventPublisher`. The `finetuning.training_step` and `finetuning.job_completed` events are never emitted from real training runs, so the metering pipeline never fires and credits are never deducted for actual GPU usage.

### 6. Checkpoint Weights Unmanaged
TT-Train writes `.pkl` checkpoint files to local filesystem. The console has S3-backed checkpoint storage with pre-signed upload/download URL flows. These are not connected — checkpoint weights are inaccessible to users via the console.

### 7. No Real-Time Bridge
TT-Train job events (progress, completion) don't flow to the console's Redis pub/sub (`sse:org:{org_id}`), so the browser's live progress view is not updated during actual training.

---

## Proposed Responsibility Split

The core principle: **TT-Train owns the compute interface; the console owns the platform.**

```
┌──────────────────────────────────────────────────────────────────────┐
│              TT-Cloud-Console (the platform layer)                   │
│                                                                      │
│  Identity & Auth    Billing & Metering    Dashboard UI               │
│  API token issuance  Credit balance        Next.js / SSE             │
│  Org / RBAC          Transaction ledger    Job history view          │
│  Session cookies     Pre-auth / settlement Metrics charts            │
│                      Stripe                                          │
│                                                                      │
│  Fine-tuning platform state:                                         │
│  fine_tuning_jobs    training_metrics       training_checkpoints      │
│  datasets (S3)       training_logs          S3 object storage        │
└──────────────────────────────────┬───────────────────────────────────┘
                                   │  HTTP (TTTrainService)
                                   │  X-TT-Organization: orgID
                                   │  Bearer: console-issued API token
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│              TT-Train (the compute engine + SDK)                     │
│                                                                      │
│  Python SDK            Cluster scheduling      ttml/ttnn integration │
│  (tt_train package)    (asyncio or Slurm)       (the unique part)    │
│                                                                      │
│  HTTP API              Worker processes         Model catalog        │
│  (auth via console)    job_runner.py            tt://catalog/...     │
│                        session_worker.py                             │
│                                                                      │
│  Interactive Sessions API (forward_backward / step / sample / eval) │
│  — this has no equivalent in the console and lives entirely here    │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Specific Interface Decisions

### Authentication

**Decision:** TT-Train validates tokens via the console.

The console's `api_tokens` table is the authority. Two options:

**Option A — Validation endpoint (preferred)**
Console exposes: `GET /v1/tokens/validate` (internal endpoint, not user-facing)
- TT-Train calls this on first use of a token, caches result with a short TTL (e.g. 5 min)
- Returns `{ org_id, user_id, scopes }` or 401
- Revocations propagate within the cache TTL

**Option B — Shared PostgreSQL**
TT-Train reads directly from the console's `api_tokens` table. Tighter coupling but no extra network hop.

Either way, TT-Train drops its current "accept anything non-empty" logic.

---

### Job Lifecycle: Shared Job ID

**Decision:** The console creates the canonical job record; TT-Train tracks execution state only.

Flow:
1. Console creates `fine_tuning_jobs` row (gets UUID `job_id`), deducts pre-auth
2. Console calls `POST /v1/jobs` on TT-Train, passing `job_id` (console's UUID) in the request body
3. TT-Train stores `console_job_id` on its own jobs row; uses it for all callbacks
4. Workers use `console_job_id` when calling back to both systems
5. TT-Train's own jobs table becomes the execution ledger (scheduler state, worker pod URLs, step count) — NOT the source of truth for billing or UI display

This means TT-Train's SDK `jobs.create()` still works directly (no console required), but when used via the console, the IDs are aligned.

---

### Worker Callbacks: Unify on Console Endpoints

**Decision:** Workers call the console's JWT-authenticated worker write endpoints for metrics, logs, and checkpoints. They continue calling TT-Train's `/internal/` for status transitions (running/complete/fail), which TT-Train then relays to the console.

Worker callback flow after this change:

```
Worker
  │── POST /internal/jobs/{id}/progress   → TT-Train (status + relay to console via webhook/SSE)
  │── POST /v1/fine-tuning/jobs/{id}/metrics         → Console (training_metrics table)
  │── POST /v1/fine-tuning/jobs/{id}/logs             → Console (training_logs table)
  │── POST /v1/fine-tuning/jobs/{id}/checkpoints/upload-url → Console → S3 pre-signed URL
  │── [PUT checkpoint to S3]
  └── POST /v1/fine-tuning/jobs/{id}/checkpoints     → Console (training_checkpoints table)
```

Worker authentication for console callbacks: the `worker_token` JWT (already issued by the console in `CreateJob` response and passed through to TT-Train for forwarding to the worker).

The `job_runner.py` `JobReporter` class needs to grow console callback support. The relevant fields to add to `JobReporter`:
- `console_base_url`
- `worker_token` (the JWT)

---

### Datasets: Console as Authority, TT-Train as Consumer

**Decision:** Datasets live in the console's S3 + `datasets` table. TT-Train workers download from signed URLs.

Flow:
1. User uploads dataset via console dashboard → stored in S3, `datasets` table tracks metadata
2. When creating a job, `dataset_url` in the request to TT-Train is a **console-issued pre-signed S3 download URL** (or a stable console API URL that redirects to a fresh pre-signed URL)
3. TT-Train's `job_runner.py` downloads the dataset from the URL to local worker temp storage at job start, then deletes it after training
4. TT-Train's own dataset management (upload endpoint, local storage) is for **direct SDK use only** (no console) — it coexists but is not the primary path

This eliminates the need for TT-Train to have shared filesystem access for dataset storage.

---

### Checkpoints: Workers Upload to S3 via Console

**Decision:** Checkpoint weights go to S3. The console issues pre-signed PUT URLs; workers upload directly.

Flow already designed in the console (`handleCheckpointUploadURL`). What needs to change in TT-Train:
- `TelemetrySFTTrainer._save_checkpoint()` calls `console_base_url + /checkpoints/upload-url` to get a PUT URL, uploads the `.pkl`, then calls `/checkpoints` to register metadata
- Workers stop writing to local `{storage_path}/checkpoints/` for anything the console tracks (can still keep local copy for resume)

---

### Billing: Workers Emit Events

**Decision:** Workers call the console's metering service directly to trigger credit deduction.

`JobReporter` (in `job_runner.py`) gets a `metering_url` and calls the `EventPublisher`-compatible endpoint on each checkpoint:

```python
# At each checkpoint / on completion:
POST {metering_url}/v1/events
Authorization: Bearer {worker_token}  # JWT, same one used for worker callbacks
{
  "event_id": "...",
  "event_type": "finetuning.training_step",
  "service_id": "tt-train",
  "status": "completed",
  "usage": {
    "model": "tt://catalog/llama-3.2-8b",
    "job_id": "{console_job_id}",
    "training_tokens": 512000,
    "gpu_seconds": 120
  }
}
```

The metering service validates the JWT, calculates cost using `model_pricing.cpu_price_per_sec`, calls `billing.DeductBalance()`, inserts a `transactions` row.

---

### SSE Real-Time Updates

**Decision:** TT-Train publishes to the console's Redis pub/sub on job status changes.

When `TTTrainService` calls TT-Train, TT-Train needs a way to push status events back to the console's SSE stream. Options:

**Option A — Webhook callback URL**
Console passes `callback_url` when creating a job on TT-Train. TT-Train posts `{ job_id, status, progress }` to it on each status change. Console's `/internal/fine-tuning/callback` handler publishes to Redis pub/sub.

**Option B — TT-Train calls console's `/internal/` endpoint directly**
Similar to the existing webhook design, but using the console's existing `/internal/jobs/{id}/progress` endpoints (rename them from TT-Train internal to console internal).

Option A is cleaner — no console-specific coupling in TT-Train.

---

### Interactive Sessions: TT-Train Only

Sessions (`forward_backward`, `step`, `sample`, etc.) have no equivalent in the console. They are billed as a separate product — charge by GPU-hour while a session is alive.

The console can expose a simple **Sessions tab** in the dashboard that:
- Creates a session via TT-Train API
- Shows session status and cost
- Lets users close a session

Billing for sessions: the session worker emits heartbeats with GPU-seconds to the metering service (same pattern as job billing).

---

## Technology Stack Alignment

| Concern | Current | Recommended |
|---|---|---|
| TT-Train scheduler | Python asyncio loop | Keep for Local/Slurm. Could adopt Temporal later but not now. |
| TT-Train DB | SQLite (dev) / PostgreSQL (prod) | Keep — execution state only. Don't merge with console DB. |
| Auth tokens | TT-Train accepts any bearer | TT-Train validates via console token endpoint |
| Dataset storage | TT-Train local filesystem | S3 (same bucket as console), download URL from console |
| Checkpoint storage | TT-Train local filesystem | S3 via console pre-signed URLs |
| Billing events | Not emitted | Workers emit to console metering service |
| SDK language | Python | Keep — agents and ML engineers use Python |

The Go vs Python split is correct. The console is the right place for Go (high-concurrency API, billing logic). TT-Train is the right place for Python (ttml bindings, ML research users, agent SDK).

---

## What Stays Where (Summary)

### Stays in TT-Train
- Python SDK (`tt_train` package) — **the primary interface for users and agents**
- Training execution: `job_runner.py`, `session_worker.py`, ttml/ttnn integration
- Interactive Sessions API (no equivalent in console)
- Model catalog (`tt://catalog/...` resolution to HuggingFace repos + ttml configs)
- Cluster scheduling (asyncio loop, LocalBackend, SlurmBackend)
- HTTP API that the SDK and console's `TTTrainService` both call
- Execution-only job state (scheduler status, worker URL, step count)

### Stays in Console
- Identity, auth, API token issuance
- Billing: credit balance, Stripe, transaction ledger, metering
- Dashboard UI (fine-tuning list, job detail, metrics charts, storage page)
- Fine-tuning job record (billing view: estimated/actual cost, pre-auth, settlement)
- `training_metrics`, `training_logs`, `training_checkpoints` tables
- `datasets` table + S3 storage for training data
- SSE real-time updates to the browser
- Admin console

### Moves / Changes Needed

| Item | From | To | Notes |
|---|---|---|---|
| Token validation | TT-Train (none) | Console `GET /v1/tokens/validate` | New endpoint on console |
| Worker metrics/logs | TT-Train DB | Console `training_metrics/logs` | Workers call console JWT endpoints |
| Worker checkpoints | Local filesystem | S3 via console | Workers use pre-signed PUT URLs |
| Billing events | Not implemented | Console metering service | Workers emit `finetuning.training_step` |
| Dataset access | Local JSONL | S3 download URL | Console provides URL when submitting job |
| SSE bridge | Not connected | Console Redis pub/sub | Via callback URL pattern |
| `console_job_id` | Not tracked | TT-Train stores it | Shared key for cross-system correlation |

---

## Migration Path

**Phase 1 — Wire auth (unblocks everything else)**
- Console adds `GET /v1/tokens/validate`
- TT-Train validates all requests against it, caches results

**Phase 2 — Align job IDs and worker callbacks**
- Console passes `console_job_id` to TT-Train on job creation
- Workers get `console_base_url` + `worker_token` + `console_job_id` in their launch args
- Workers call console's JWT endpoints for metrics/logs/checkpoints

**Phase 3 — Dataset and checkpoint S3 integration**
- Console provides dataset download URL in job creation request
- Workers download dataset at job start
- Workers request checkpoint upload URLs from console, upload to S3

**Phase 4 — Billing**
- Workers emit `finetuning.training_step` events to metering service
- Console settles actual cost on job completion/failure

**Phase 5 — SSE bridge**
- TT-Train calls console callback URL on job status transitions
- Console publishes to Redis → browser receives live updates
