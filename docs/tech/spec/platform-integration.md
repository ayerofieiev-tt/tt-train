# Platform ↔ Training Service: Integration Contract

**Scope:** The 7 endpoints the Platform actually calls. Everything else in the Training Service API is SDK/standalone-facing and out of scope here.

---

## Changes Required

### Platform side
- Add `Authorization: Bearer {tt_train_api_key}` header to all requests
- Add `method` field to job creation (required by Training Service)
- Rename `training_params` → `config`; move `optimizer` inside `config`
- Map `cluster_size` string → `hardware.nodes` int (see mapping below)
- Add `console_job_id` to job creation (Platform's own job ID, for correlation)
- Read `error.message` instead of top-level `error_message`
- Read `model_path` instead of `s3_url` on checkpoints
- Read `log_type` / `logged_at` instead of `type` / `timestamp` on logs

### Training Service side
- Accept `X-TT-Organization` header and store on job row (for org-scoped listing)
- Add `estimated_cost_cents: int` and `actual_cost_cents: int | null` to job response
- Add `cluster_size` mapping: resolve `hardware.nodes` → `"N_cards"` string if Platform still needs it (or drop from Platform)
- Remove `console_base_url`, `worker_token`, `callback_url` from job creation (Platform does not use these)

---

## Endpoint Contracts

### `POST /v1/jobs`

**Request**
```json
{
  "model":        "string (required)",
  "method":       "sft | dpo | rl (required)",
  "dataset_url":  "string (required — Platform always uses URL-based datasets)",
  "config": {
    "learning_rate": 2e-5,
    "epochs":        3,
    "batch_size":    4,
    "optimizer":     "adamw"
  },
  "hardware": {
    "accelerator": "wormhole",
    "nodes":       1
  },
  "console_job_id": "string (Platform's own job UUID, stored for correlation)"
}
```

`cluster_size` → `hardware.nodes` mapping:
| Platform `cluster_size` | Training Service `hardware.nodes` |
|---|---|
| `"1_card"` | 1 |
| `"4_cards"` | 1 (4 devices per node) |
| `"8_cards"` | 1 |
| `"16_cards"` | 2 |
| `"32_cards"` | 4 |

**Response**
```json
{
  "id":                   "job_xxxxx",
  "status":               "queued | running | completed | failed | cancelled",
  "model":                "string",
  "method":               "string",
  "dataset_url":          "string",
  "config":               {},
  "hardware":             {},
  "error": {
    "type":    "string",
    "message": "string"
  },
  "result_model":         "string | null",
  "estimated_cost_cents": 2550,
  "actual_cost_cents":    null,
  "created_at":           "ISO8601",
  "started_at":           "ISO8601 | null",
  "completed_at":         "ISO8601 | null",
  "console_job_id":       "string | null"
}
```

---

### `GET /v1/jobs`

**Request headers**
```
Authorization:    Bearer {tt_train_api_key}
X-TT-Organization: {org_id}
```

**Query params:** `limit`, `after`, `status`

**Response** — paginated list of job objects (same shape as above)

```json
{
  "object":   "list",
  "data":     [ ... ],
  "has_more": true,
  "first_id": "job_xxx",
  "last_id":  "job_yyy"
}
```

Training Service must filter by `X-TT-Organization` when present.

---

### `GET /v1/jobs/{id}`

**Response** — single job object (same shape as above)

---

### `POST /v1/jobs/{id}/cancel`

No request body.

**Response** — job object with `status: "cancelled"`

---

### `GET /v1/jobs/{id}/metrics`

**Response**
```json
{
  "object": "list",
  "job_id": "string",
  "data": [
    {
      "step":              100,
      "epoch":             1.0,
      "train_loss":        2.45,
      "val_loss":          2.61,
      "grad_norm":         0.92,
      "learning_rate":     2e-5,
      "tokens_per_second": 1200.0,
      "recorded_at":       "ISO8601"
    }
  ]
}
```

---

### `GET /v1/jobs/{id}/logs`

**Response**
```json
{
  "object": "list",
  "job_id": "string",
  "has_more": false,
  "data": [
    {
      "id":        "log_xxx",
      "step":      100,
      "log_type":  "info | warning | error | checkpoint | eval",
      "message":   "string",
      "logged_at": "ISO8601"
    }
  ]
}
```

Platform adapts to `log_type` / `logged_at` (Training Service field names win).

---

### `GET /v1/jobs/{id}/checkpoints`

**Response**
```json
{
  "object":   "list",
  "data": [
    {
      "id":         "ckpt_xxx",
      "step":       500,
      "epoch":      1.0,
      "metrics": {
        "train_loss": 2.1,
        "val_loss":   2.3
      },
      "model_path": "tt://checkpoints/job_xxx/step_500.pkl",
      "name":       "string | null",
      "created_at": "ISO8601"
    }
  ]
}
```

Platform adapts to `model_path` (Training Service field name wins). `s3_url`, `size_bytes`, `ckpt_type` are not provided — Platform drops those expectations or requests them later as an extension.

---

## Out of Scope for Platform Integration

The following Training Service endpoints are **not called by the Platform** and serve SDK/standalone users only:

- `POST|GET /v1/sessions/**` — interactive training (Platform has no UI for this)
- `POST|GET /v1/datasets/**` — Platform manages its own dataset storage
- `GET /v1/models/**` — Platform has its own model catalog
- `GET /v1/hardware/catalog` — Platform hardcodes hardware options
- `POST /v1/jobs/estimate` — Platform may add this later
- `POST /v1/jobs/{id}/pause|resume` — Platform may add these later

---

## Telemetry Events

The Training Service worker emits events to the Platform's metering endpoint. The Platform publishes these onward to Kafka — this is a **one-way push from Training Service to Platform**, not a callback.

See `training-service-interface.md` §"Telemetry Events" for the event schemas (`finetuning.training_step`, `finetuning.job_completed`).

**Current state:** Platform's `publisher.go` emits these itself. Decision needed: does the Training Service worker emit them, or does the Platform emit them after polling? One side should own this — not both.
