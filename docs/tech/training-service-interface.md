# Training Service Interface

**Version:** v1
**Owner:** Platform (TT-Cloud-Console)
**Status:** Authoritative

This document is the Platform's specification for any Training Service it integrates with. The Training Service must implement everything defined here. The Platform is the source of truth for this contract — changes originate there, and the Training Service updates to comply.

- **Platform** — owns this spec; handles identity, storage, and the user-facing dashboard; calls the Training Service on behalf of users
- **Training Service** — implements this spec; executes training workloads, manages sessions, owns training state (metrics, logs, checkpoints)

---

## Deployment Model

The Training Service operates in two modes:

**Managed** — deployed as an internal service behind the Platform's API gateway. The Platform validates user tokens before forwarding; the Training Service trusts the `X-Organization` header unconditionally and does not need to perform token validation itself. The Training Service is not on the public internet.

**Standalone** — deployed independently, accessed directly by SDK users or agents without a Platform. Authentication is handled by a configured API key. No SSE bridge, no Platform callbacks.

The interfaces defined below cover both modes. Platform-integration fields are all optional; omitting them activates standalone behavior.

---

## Authentication

### Platform → Training Service

Every request from the Platform carries two headers:

```
Authorization: Bearer <api-token>
X-Organization: <org-uuid>
```

In managed mode, the Platform has already validated the token. The Training Service trusts `X-Organization` as the authoritative org scope.

In standalone mode, the Training Service validates the Bearer token against its own configured API key.

### Workers → Platform

Workers use a short-lived job-scoped JWT issued by the Platform at job creation time, passed through the Training Service to the worker process at launch.

```
Authorization: Bearer <worker_token>
```

JWT claims:
```json
{
  "org_id":  "<org-uuid>",
  "user_id": "<user-uuid>",
  "job_id":  "<platform-job-uuid>",
  "exp":     <unix timestamp, 7 days from issuance>
}
```

---

## Platform → Training Service: REST API

Base URL: configured on the Platform side (e.g. `TRAINING_SERVICE_URL`).

All requests include `X-Organization` and `Authorization` headers.

### Jobs

#### Create Job

```
POST /v1/jobs
```

Request:
```json
{
  "model":       "catalog/llama-3.2-8b",
  "method":      "sft",
  "dataset_url": "https://storage.example.com/datasets/ds_abc/data.jsonl?signed=...",
  "config": {
    "max_steps":                   1000,
    "lr":                          2e-5,
    "batch_size":                  4,
    "gradient_accumulation_steps": 1,
    "max_seq_len":                 1024,
    "warmup_steps":                50,
    "seed":                        42
  },
  "hardware": {
    "accelerator": "wormhole",
    "nodes":       1
  },

  "platform_job_id":  "<platform's canonical UUID for this job>",
  "platform_base_url": "https://api.example.com",
  "worker_token":      "<job-scoped JWT>",
  "callback_url":      "https://api.example.com/internal/training/callback",

  "name":     "my-finetune",
  "metadata": {}
}
```

Required fields: `model`, `method`, and at least one of `dataset_url` or `training_data`.

Platform-integration fields (`platform_job_id`, `platform_base_url`, `worker_token`, `callback_url`) are all optional. When omitted the service runs in standalone mode.

Response `201 Created`:
```json
{
  "object":          "job",
  "id":              "<training-service-job-id>",
  "platform_job_id": "<echoed back>",
  "dataset_url":     "<echoed back>",
  "model":           "catalog/llama-3.2-8b",
  "method":          "sft",
  "status":          "queued",
  "config":          {},
  "hardware":        {},
  "progress":        null,
  "metrics":         null,
  "result_model":    null,
  "error":           null,
  "cost":            { "estimated_total": "$12.40", "estimated_time_seconds": 3600 },
  "created_at":      "2026-03-06T12:00:00Z",
  "started_at":      null,
  "completed_at":    null
}
```

`worker_token`, `platform_base_url`, and `callback_url` are **not** echoed in the response — they are write-only fields used for worker dispatch.

#### Get Job

```
GET /v1/jobs/{id}
```

Response: same shape as Create Job response.

`status` values: `queued` | `running` | `paused` | `completed` | `failed` | `cancelled`

#### Cancel Job

```
POST /v1/jobs/{id}/cancel
```

Response: updated job object with `status: "cancelled"`.

#### Estimate Cost

```
POST /v1/jobs/estimate
```

Request: same fields as Create Job. No side effects.

Response:
```json
{
  "object":                 "estimate",
  "estimated_cost":         "$12.40",
  "estimated_time_seconds": 3600,
  "estimated_steps":        1000,
  "tokens_total":           1000000,
  "hardware_plan": {
    "accelerator": "wormhole",
    "nodes":       1
  },
  "cost_breakdown": {
    "compute":       "$11.90",
    "storage":       "$0.50",
    "data_transfer": "$0.00"
  }
}
```

#### Get Training Metrics

```
GET /v1/jobs/{id}/metrics
```

Returns per-step training metrics. The Platform reads this to render loss charts and metrics timeseries in the dashboard.

Response:
```json
{
  "object":   "list",
  "job_id":   "<id>",
  "has_more": false,
  "data": [
    {
      "step":              100,
      "epoch":             0.1,
      "train_loss":        2.341,
      "val_loss":          2.512,
      "grad_norm":         0.843,
      "learning_rate":     2e-5,
      "tokens_per_second": 1240.5,
      "recorded_at":       "2026-03-06T12:05:00Z"
    }
  ]
}
```

Ordered by `step` ascending.

#### Get Training Logs

```
GET /v1/jobs/{id}/logs?log_type=<type>&limit=<n>
```

Query params:
- `log_type` (optional): `info` | `warning` | `error` | `checkpoint` | `eval`
- `limit` (optional, default 200, max 1000)

Response:
```json
{
  "object":   "list",
  "job_id":   "<id>",
  "has_more": false,
  "data": [
    {
      "id":        "<uuid>",
      "step":      100,
      "log_type":  "checkpoint",
      "message":   "Saved checkpoint at step 100, loss=2.341",
      "logged_at": "2026-03-06T12:05:00Z"
    }
  ]
}
```

#### Get Checkpoints

```
GET /v1/jobs/{id}/checkpoints
```

Response:
```json
{
  "object":   "list",
  "has_more": false,
  "data": [
    {
      "object":     "checkpoint",
      "id":         "<id>",
      "job_id":     "<job-id>",
      "model_path": "checkpoints/<job-id>/step_1000",
      "step":       1000,
      "epoch":      1.0,
      "metrics":    { "train_loss": 1.23 },
      "created_at": "2026-03-06T13:00:00Z"
    }
  ]
}
```

---

### Sessions

Sessions are long-lived interactive training handles. A session worker loads the model into accelerator memory and stays alive, accepting commands until explicitly closed or idle-expired.

#### Create Session

```
POST /v1/sessions
```

Request:
```json
{
  "model": "catalog/llama-3.2-8b",
  "optimizer": {
    "type":         "adamw",
    "lr":           2e-5,
    "weight_decay": 0.01
  },
  "lora": {
    "r":              16,
    "alpha":          32,
    "target_modules": ["q_proj", "v_proj"]
  },
  "hardware": {
    "accelerator": "wormhole",
    "nodes":       1
  },
  "idle_timeout_minutes": 30,
  "name":     "rl-experiment-1",
  "metadata": {}
}
```

Response `201 Created`:
```json
{
  "object":               "session",
  "id":                   "<id>",
  "model":                "catalog/llama-3.2-8b",
  "status":               "provisioning",
  "step_count":           0,
  "idle_timeout_minutes": 30,
  "last_checkpoint":      null,
  "created_at":           "2026-03-06T12:00:00Z",
  "expires_at":           null,
  "closed_at":            null
}
```

`status` values: `provisioning` | `ready` | `closed` | `expired` | `failed`

The session transitions from `provisioning` to `ready` once the worker has loaded the model and is accepting commands. Callers poll `GET /v1/sessions/{id}` until `status == "ready"`.

#### Get / List / Close Session

```
GET    /v1/sessions/{id}
GET    /v1/sessions?status=<status>&limit=<n>
DELETE /v1/sessions/{id}
```

`DELETE` triggers a graceful worker shutdown. Billing stops at close time.

#### Training Commands

All commands are synchronous — the caller blocks until the operation completes. The worker processes one command at a time per session, which preserves gradient state correctness.

Default timeouts: 300s for `forward_backward` and `step`; 600s for `sample` and `eval`. Implementations may make these configurable.

```
POST /v1/sessions/{id}/forward_backward
```
```json
{
  "batch": [
    { "messages": [
      { "role": "user",      "content": "What is 2+2?" },
      { "role": "assistant", "content": "4" }
    ]}
  ],
  "loss": "cross_entropy"
}
```
Response: `{ "loss": 2.341, "token_count": 128, "example_count": 1, "grad_norm": null, "duration_ms": 380 }`

---

```
POST /v1/sessions/{id}/step
```
```json
{ "max_grad_norm": 1.0 }
```
Response: `{ "step_number": 1, "learning_rate": 2e-5, "grad_norm_before_clip": 0.84, "grad_norm_after_clip": 0.84, "duration_ms": 25 }`

---

```
POST /v1/sessions/{id}/sample
```
```json
{
  "prompts":          [{ "messages": [{ "role": "user", "content": "Hello" }] }],
  "temperature":      0.8,
  "top_p":            0.95,
  "max_tokens":       256,
  "n":                1,
  "return_log_probs": false
}
```
Response: `{ "completions": [{ "prompt_index": 0, "outputs": [{ "index": 0, "text": "...", "tokens": 42, "finish_reason": "stop" }] }], "usage": { "prompt_tokens": 10, "completion_tokens": 42 } }`

---

```
POST /v1/sessions/{id}/log_probs
```
```json
{
  "batch": [
    { "prompt": "What is 2+2?", "completion": "4" }
  ]
}
```
Response: `{ "scores": [{ "index": 0, "total_log_prob": -0.23, "avg_log_prob": -0.23, "tokens": 1, "per_token": [] }] }`

---

```
POST /v1/sessions/{id}/eval
```
```json
{
  "data":         [{ "messages": [...] }],
  "metrics":      ["loss", "perplexity"],
  "max_examples": 100,
  "batch_size":   8
}
```
Response: `{ "examples_evaluated": 100, "metrics": { "loss": 1.82, "perplexity": 6.17 }, "duration_ms": 4200 }`

---

```
POST /v1/sessions/{id}/save
```
```json
{
  "name":     "checkpoint-after-episode-5",
  "metadata": { "episode": 5, "reward": 0.82 }
}
```
Response: `{ "object": "checkpoint", "id": "<id>", "model_path": "checkpoints/...", "step": 42, ... }`

---

### Catalog and Hardware

```
GET /v1/models     # available base models
GET /v1/hardware   # available hardware and current capacity
```

Informational — the Platform uses these to populate the job creation UI.

---

## Training Service → Platform: Callbacks

### Job Status Callback

On transition to `running`, `completed`, or `failed`, the Training Service POSTs to the `callback_url` supplied at job creation.

```
POST {callback_url}
Authorization: Bearer {worker_token}
Content-Type: application/json
```

**Running:**
```json
{
  "platform_job_id": "<uuid>",
  "status":          "running",
  "started_at":      "2026-03-06T12:01:00Z"
}
```

**Completed:**
```json
{
  "platform_job_id": "<uuid>",
  "status":          "completed",
  "result_model":    "checkpoints/<job-id>/final",
  "completed_at":    "2026-03-06T13:05:00Z"
}
```

**Failed:**
```json
{
  "platform_job_id": "<uuid>",
  "status":          "failed",
  "error": {
    "type":    "runtime_error",
    "message": "Out of memory"
  }
}
```

The Platform's callback handler must:
1. Verify the `worker_token` JWT
2. Update the job's status record
3. Publish the status change to connected browser clients (e.g. via Redis pub/sub → SSE)

The Training Service fires callbacks once and logs failures without retrying. The Platform must not rely on callbacks as the sole source of truth — it should reconcile against `GET /v1/jobs/{id}` for any missed transitions.

---

## Training Service → Platform: Telemetry Events

The Training Service emits structured events during and after training. These make runs observable and auditable. What the Platform does with them is outside the Training Service's concern.

### Pipeline

```
Training Service worker
  │
  └─► POST {platform_base_url}/v1/events      (Platform metering endpoint)
        │  validates envelope + usage schema
        │  extracts org_id, user_id from JWT (not from body)
        │
        └─► Kafka topic: finetuning.*          (keyed by org_id for ordered processing)
              │
              └─► Kafka consumer
```

### Authentication

```
POST {platform_base_url}/v1/events
Authorization: Bearer {worker_token}
Content-Type: application/json
```

The `worker_token` JWT carries `org_id` and `user_id`. The metering service extracts these to enrich the Kafka message. Identity does not belong in the event body.

### Event Envelope

All events share the same envelope:

```json
{
  "event_id":   "<uuid v4, must be unique per event>",
  "event_type": "<type string>",
  "service_id": "tt-train",
  "timestamp":  "<ISO 8601, optional — server fills in if omitted>",
  "status":     "completed | failed | cancelled",
  "usage":      { ... }
}
```

Required fields: `event_id`, `event_type`, `service_id`, `usage`. Events with unknown `event_type` values are rejected `400`. The Platform endpoint is idempotent on `event_id`.

`status` reflects the outcome of the training work being reported:
- `completed` — step or job finished successfully
- `failed` — job failed; emit with totals accumulated up to the point of failure
- `cancelled` — job was cancelled; emit with totals up to the cancellation point

### `finetuning.training_step`

Emitted at each checkpoint during training.

```json
{
  "event_id":   "<uuid>",
  "event_type": "finetuning.training_step",
  "service_id": "tt-train",
  "status":     "completed",
  "usage": {
    "model":           "catalog/llama-3.2-8b",
    "job_id":          "<platform_job_id>",
    "training_tokens": 512000,
    "epoch":           1,
    "total_epochs":    3,
    "tpu_seconds":     120
  }
}
```

| Field | Required | Constraint |
|---|---|---|
| `model` | yes | non-empty string |
| `job_id` | yes | non-empty string — the Platform's canonical job UUID |
| `training_tokens` | yes | integer > 0, ≤ 10,000,000,000 |
| `epoch` | no | current epoch, informational |
| `total_epochs` | no | informational |
| `tpu_seconds` | no | TPU time since last event |

### `finetuning.job_completed`

Emitted once when a job reaches a terminal state (`completed`, `failed`, or `cancelled`). Carries totals for the full run.

```json
{
  "event_id":   "<uuid>",
  "event_type": "finetuning.job_completed",
  "service_id": "tt-train",
  "status":     "completed | failed | cancelled",
  "usage": {
    "model":                 "catalog/llama-3.2-8b",
    "job_id":                "<platform_job_id>",
    "total_training_tokens": 5120000,
    "epochs":                3,
    "total_tpu_seconds":     1200
  }
}
```

| Field | Required | Constraint |
|---|---|---|
| `model` | yes | non-empty string |
| `job_id` | yes | non-empty string |
| `total_training_tokens` | yes | integer > 0, ≤ 10,000,000,000 |
| `epochs` | no | informational |
| `total_tpu_seconds` | no | informational |

Emit this event regardless of terminal status. A failed or cancelled job that processed some tokens must still report the actual totals.

### Delivery semantics

Workers emit on a best-effort basis without retry. The Platform endpoint must handle duplicates via `event_id` idempotency. Missing `training_step` events are tolerable — the `job_completed` event provides totals for reconciliation.

> **Schema compatibility:** The Platform's event consumer must stay in sync with the field names defined here. The Platform's current `events/types.go` uses `gpu_seconds` / `total_gpu_seconds` — these must be renamed to `tpu_seconds` / `total_tpu_seconds` across `types.go`, the validator, and the Kafka consumer.

---

## Error Responses

```json
{
  "error": {
    "type":    "<error_type>",
    "message": "<human-readable description>",
    "id":      "<resource id, if applicable>"
  }
}
```

| HTTP status | `type` |
|---|---|
| 400 | `invalid_request_error` |
| 401 | `authentication_error` |
| 403 | `permission_denied_error` |
| 404 | `not_found_error` |
| 409 | `conflict_error` |
| 503 | `session_not_ready` |
| 5xx | `api_error` |

---

## Versioning

All endpoints are under `/v1/`. Breaking changes require a new version prefix with a deprecation period. Non-breaking additions (new optional fields) may be made without a version bump. Both sides must tolerate unknown fields in responses.

---

## Responsibility Boundary

### Platform owns

- User identity, org management, API token issuance
- Dataset storage — object store + metadata; provides a download URL when creating a job
- Checkpoint object storage — issues pre-signed upload/download URLs on request
- SSE delivery to browser clients
- The canonical job record (`platform_job_id` row: status, estimated/actual cost, user-facing state)

### Training Service owns

- Training execution and hardware scheduling
- `training_metrics`, `training_logs`, `checkpoints` — the Platform reads these via the API above
- Session worker lifecycle: provisioning, registration, idle expiry, command proxying
- Model catalog: resolution of model identifiers to weights and configurations
- Interactive Sessions API — there is no Platform equivalent; this is training-service-only functionality
