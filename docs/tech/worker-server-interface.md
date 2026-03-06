# Worker / Server Interface

**Version:** v1
**Status:** Authoritative contract

This document defines the two-sided interface between **workers** (processes running on compute nodes) and the **Training Service** (the API server).

There are two distinct channels:

1. **Worker → Server (callbacks):** Workers POST to the server's internal API to report lifecycle events, progress, and logs. This is the server's `POST /v1/internal/*` surface.
2. **Server → Session Worker (command proxy):** The server forwards session training commands to a session worker's own HTTP server. The session worker owns its HTTP surface; the server is the client.

Both sides must comply. Changes to any endpoint or payload shape defined here require coordinated updates.

---

## Authentication

### Worker → Server callbacks

Workers authenticate using the internal API key:

```
Authorization: Bearer <settings.internal_api_key>
```

The server validates this against `TT_TRAIN_INTERNAL_API_KEY` (default: `"internal-secret"`). Any mismatch returns `401`.

This key is passed to workers at launch via `--api-key`. It is distinct from the user-facing API key.

### Server → Session Worker

The server makes unauthenticated HTTP requests to the session worker's local HTTP server. The session worker is not on the public network — it binds on the compute node and its URL (`worker_url`) is only stored server-side. No auth header is sent.

---

## Worker Launch Arguments

Workers are invoked as subprocesses (or Slurm batch jobs). All configuration is passed as CLI arguments; workers do not read environment variables for functional configuration.

### `job_runner.py`

```
python workers/job_runner.py \
  --job-id          <job.id>               [required]
  --api-url         <TT_TRAIN_API_BASE_URL> [required]
  --api-key         <TT_TRAIN_INTERNAL_API_KEY> [required]
  --model           <job.model>            [required]
  --method          <job.method>           [required]
  --training-data   <job.training_data>    [optional]
  --validation-data <job.validation_data>  [optional]
  --config          <json blob>            [optional, default "{}"]
  --storage-path    <shared_storage_path>  [optional, default /tmp/tt_train_storage]
  --dataset-url     <presigned URL>        [optional — mutually exclusive with --training-data]
  --console-job-id  <platform job UUID>    [optional — console integration]
  --console-base-url <platform base URL>  [optional — console integration]
  --worker-token    <job-scoped JWT>       [optional — console integration]
  --callback-url    <platform callback URL> [optional — console integration]
```

When `--dataset-url` is provided, the runner downloads the dataset to `{storage_path}/datasets/{job_id}/data.jsonl` before training begins. When omitted, it reads from `{storage_path}/datasets/{training_data}/data.jsonl`.

### `session_worker.py`

```
python workers/session_worker.py \
  --session-id       <session.id>              [required]
  --api-url          <TT_TRAIN_API_BASE_URL>   [required]
  --api-key          <TT_TRAIN_INTERNAL_API_KEY> [required]
  --model            <session.model>           [required]
  --lora-config      <json blob>               [optional, default "{}"]
  --optimizer-config <json blob>               [optional, default "{}"]
  --storage-path     <shared_storage_path>     [optional, default /tmp/tt_train_storage]
  --port             <int>                     [optional, default 0 (auto-select)]
```

---

## Worker → Server: Internal Callback API

Base: `{api_url}/internal/`

All requests carry `Authorization: Bearer <api_key>` and `Content-Type: application/json`.

---

### `POST /internal/jobs/{job_id}/progress`

Incremental training progress update. Called at each checkpoint save and every 30 s between checkpoints (via `ProgressPoller`).

Request:
```json
{
  "step":             100,
  "total_steps":      1000,
  "epoch":            0.1,
  "percentage":       10.0,
  "loss":             2.341,
  "val_loss":         2.512,        // optional
  "grad_norm":        0.843,        // optional
  "learning_rate":    2e-5,         // optional
  "tokens_per_second": 1240.5,      // optional
  "tokens_processed": 51200         // optional
}
```

Response `200`:
```json
{ "ok": true }
```

The server updates `job.progress` and `job.metrics`, and writes a row to `training_metrics` if `step` is present.

---

### `POST /internal/jobs/{job_id}/logs`

Append a structured log entry.

Request:
```json
{
  "log_type": "info",          // "info" | "warning" | "error" | "checkpoint" | "eval"
  "message":  "Saved checkpoint at step 100",
  "step":     100              // optional
}
```

Response `200`:
```json
{
  "id":       "<uuid>",
  "job_id":   "<job_id>",
  "log_type": "checkpoint",
  "message":  "Saved checkpoint at step 100",
  "step":     100
}
```

---

### `POST /internal/jobs/{job_id}/complete`

Signal successful job completion. Called once after training finishes.

Request:
```json
{
  "result_model": "tt://checkpoints/<job_id>/final",
  "metrics": {
    "train_loss": 1.234
  }
}
```

Response `200`:
```json
{ "ok": true }
```

The server sets `status="completed"`, stores `result_model` and `metrics`, sets `completed_at`, and creates a checkpoint record for `result_model`.

---

### `POST /internal/jobs/{job_id}/fail`

Signal job failure. Called on unhandled exception or `KeyboardInterrupt` (cancellation).

Request:
```json
{
  "type":    "runtime_error",    // "runtime_error" | "cancelled" | any string
  "message": "CUDA out of memory",
  "step":    42                  // optional — last completed step before failure
}
```

Response `200`:
```json
{ "ok": true }
```

The server sets `status="failed"`, stores `error`, sets `completed_at`.

---

### `POST /internal/sessions/{session_id}/ready`

Signal that the session worker has loaded the model and is accepting commands. Called once on startup, after the HTTP server is listening.

Request:
```json
{
  "worker_url": "http://compute-node-07:49152"
}
```

`worker_url` is required. The server rejects missing or empty values with `400`.

Response `200`:
```json
{ "ok": true }
```

The server sets `status="ready"`, stores `worker_url`, sets `last_active_at=now()`. Clients polling `GET /v1/sessions/{id}` will see `status="ready"` after this point.

---

### `POST /internal/sessions/{session_id}/heartbeat`

Keep the session alive. Prevents idle-timeout expiry. The session worker may call this at any interval shorter than `idle_timeout_minutes`.

Request body: any JSON object (ignored).

Response `200`:
```json
{ "ok": true }
```

The server sets `last_active_at=now()`.

> **Note:** The session worker does not currently send explicit heartbeats. `last_active_at` is updated by training primitive calls (forward_backward, step, etc.) via the server's `_touch()` function. Heartbeats are available for session workers that need to hold the session alive during long idle periods between user commands.

---

## Job Runner Lifecycle

```
launch
  │
  ├─ notify_callback("running")  →  POST {callback_url} with status="running"
  │                                  (only when --callback-url is set)
  │
  ├─ [download dataset if --dataset-url]
  │
  ├─ training loop
  │     ├─ on checkpoint:  POST /internal/jobs/{id}/progress
  │     │                  POST /internal/jobs/{id}/logs  (log_type="checkpoint")
  │     │                  emit_usage_event("finetuning.training_step")
  │     └─ every 30 s:     POST /internal/jobs/{id}/progress  (ProgressPoller)
  │
  ├─ on success:
  │     POST /internal/jobs/{id}/complete
  │     notify_callback("completed")
  │     emit_usage_event("finetuning.job_completed", status="completed")
  │
  └─ on failure / KeyboardInterrupt:
        POST /internal/jobs/{id}/fail
        notify_callback("failed")
        emit_usage_event("finetuning.job_completed", status="failed|cancelled")
        sys.exit(1)  (or 0 for cancellation)
```

All `_post()` calls are fire-and-forget: exceptions are logged as warnings, not re-raised. A failed progress report does not abort training.

---

## Session Worker Lifecycle

```
launch
  │
  ├─ load model (RealModelState via ttml, or SimModelState fallback)
  │
  ├─ bind HTTP server on 0.0.0.0:{port}
  │     port = --port if given, else OS-assigned free port
  │
  ├─ register_with_api():
  │     POST /internal/sessions/{id}/ready  { "worker_url": "http://{hostname}:{port}" }
  │     Retries up to 10 times with exponential backoff (max 30 s per attempt)
  │     Raises RuntimeError and exits if all attempts fail
  │
  ├─ serve requests (blocking main thread watches shutdown_flag)
  │
  └─ on /shutdown or SIGINT:
        server.shutdown()
        process exits
```

The worker uses `socket.gethostname()` as the hostname in `worker_url`. In Slurm environments this resolves to the compute node's FQDN or internal IP, which must be reachable from the API server.

---

## Server → Session Worker: Command Proxy

The server proxies all training primitive requests to the session worker without modification. There is no re-authentication — the worker trusts all requests on its local port.

**Proxy function:**

```
POST {session.worker_url}/{command}
Content-Type: application/json
Timeout: 300 s
```

The request body is forwarded verbatim. The response body is forwarded verbatim to the client.

If `session.worker_url` is not set (session not yet `ready`), the server returns `503 session_not_ready`.

After each successful proxy call the server updates `last_active_at = now()`.

---

## Session Worker HTTP API

The session worker runs a plain `http.server.HTTPServer` (not HTTPS). All routes accept `POST` with a JSON body. All responses are JSON.

### `POST /forward_backward`

Request:
```json
{
  "batch": [
    { "messages": [
      { "role": "user",      "content": "What is 2+2?" },
      { "role": "assistant", "content": "4" }
    ]}
  ],
  "loss":        "cross_entropy",
  "loss_config": {}
}
```

Response `200`:
```json
{
  "object":        "forward_backward_result",
  "session_id":    "<id>",
  "loss":          2.341,
  "token_count":   128,
  "example_count": 1,
  "grad_norm":     null,
  "duration_ms":   380
}
```

Gradients are accumulated but not applied. Multiple `forward_backward` calls before a `step` implement gradient accumulation.

---

### `POST /step`

Request:
```json
{ "max_grad_norm": 1.0 }
```

Response `200`:
```json
{
  "object":                "step_result",
  "session_id":            "<id>",
  "step_number":           1,
  "learning_rate":         2e-5,
  "grad_norm_before_clip": 0.843,
  "grad_norm_after_clip":  0.843,
  "duration_ms":           25
}
```

Applies accumulated gradients, advances the LR schedule, clears gradients. `step_number` is the new step count after this call.

---

### `POST /sample`

Request:
```json
{
  "prompts":          [{ "messages": [{ "role": "user", "content": "Hello" }] }],
  "temperature":      0.8,
  "top_p":            0.95,
  "max_tokens":       256,
  "n":                1,
  "return_log_probs": false,
  "seed":             42
}
```

Response `200`:
```json
{
  "object":      "sample_result",
  "session_id":  "<id>",
  "completions": [
    {
      "prompt_index": 0,
      "outputs": [
        { "index": 0, "text": "...", "tokens": 42, "finish_reason": "stop" }
      ]
    }
  ],
  "usage": { "prompt_tokens": 10, "completion_tokens": 42 }
}
```

`finish_reason`: `"stop"` (EOS reached) or `"length"` (max_tokens reached).

---

### `POST /log_probs`

Request:
```json
{
  "batch": [
    { "prompt": "What is 2+2?", "completion": "4" }
  ]
}
```

Response `200`:
```json
{
  "object":     "log_probs_result",
  "session_id": "<id>",
  "scores": [
    {
      "index":         0,
      "total_log_prob": -0.23,
      "avg_log_prob":   -0.23,
      "tokens":         1,
      "per_token":      []
    }
  ]
}
```

No gradient accumulation — forward pass only under `no_grad`.

---

### `POST /eval`

Request:
```json
{
  "data":         [{ "messages": [...] }],
  "metrics":      ["loss", "perplexity"],
  "max_examples": 100,
  "batch_size":   32
}
```

`data` may be a list of message-format examples or a dataset ID string (resolved to `{storage_path}/datasets/{id}/data.jsonl`).

Response `200`:
```json
{
  "object":            "eval_result",
  "session_id":        "<id>",
  "step":              42,
  "examples_evaluated": 100,
  "metrics":           { "loss": 1.82, "perplexity": 6.17 },
  "duration_ms":       4200
}
```

Supported metric keys: `loss`, `perplexity`. No gradients accumulated.

---

### `POST /save`

Request:
```json
{
  "name":     "checkpoint-after-episode-5",
  "metadata": { "episode": 5, "reward": 0.82 }
}
```

Response `200`:
```json
{
  "object":     "checkpoint",
  "id":         "ckpt_<session_suffix>_<step>",
  "model_path": "tt://checkpoints/<session_id>/<ckpt_id>",
  "session_id": "<id>",
  "step":       42,
  "name":       "checkpoint-after-episode-5",
  "metrics":    { "train_loss": 1.23 },
  "metadata":   { "episode": 5, "reward": 0.82 }
}
```

The server persists this checkpoint record into the DB and updates `session.last_checkpoint`.

---

### `POST /shutdown`

Request: any JSON object (ignored).

Response `200`:
```json
{ "status": "shutting_down" }
```

Sets a flag that causes the main loop to exit after the response is sent. The HTTP server shuts down cleanly. Called by the server on `DELETE /v1/sessions/{id}` and by the Scheduler on idle expiry (both fire-and-forget with a 10 s timeout).

---

### Error responses (session worker)

On unhandled exception the handler returns `500`:

```json
{ "error": "<exception message>" }
```

On unknown path: `404 { "error": "unknown path" }`.

The server propagates these back to the API client as-is.

---

## Shared Storage Layout

Workers read datasets and write checkpoints to `settings.shared_storage_path` (default: `/tmp/tt_train_storage`). This path must be accessible from both the API server and all compute nodes.

```
{shared_storage}/
  datasets/
    {dataset_id}/
      data.jsonl          # JSONL, each line: {"messages": [...]}
    {job_id}/             # downloaded from --dataset-url
      data.jsonl
  checkpoints/
    {job_id}/
      final/              # written by job_runner on completion
      ckpts/
        step_{n}.pkl      # intermediate checkpoints
    {session_id}/
      {ckpt_id}/          # written by session_worker on /save
        step_{n}.pkl
```

---

## Model Catalog

Model identifiers are resolved via `workers/common.py:MODEL_CATALOG`. The catalog maps `tt://catalog/*` IDs to HuggingFace repos and ttml config paths. Both `job_runner.py` and `session_worker.py` use this catalog. Unknown model IDs raise `ValueError` at launch.

| ID | HuggingFace repo |
|---|---|
| `tt://catalog/llama-3.2-8b` | `meta-llama/Llama-3.2-8B` |
| `tt://catalog/llama-3.2-1b` | `meta-llama/Llama-3.2-1B-Instruct` |
| `tt://catalog/llama-3.1-70b` | `meta-llama/Llama-3.1-70B` |
| `tt://catalog/tinyllama` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| `tt://catalog/mistral-7b-v0.3` | `mistralai/Mistral-7B-v0.3` |

---

## Simulation Fallback

Both workers detect whether `ttml`/`ttnn` are importable at startup. If not, they fall back to simulation mode (`run_training_sim` / `SimModelState`) with no TT hardware dependency. The callback API surface and lifecycle are identical in both modes.

---

## Responsibility Boundary

### Server owns

- Routing user API calls to the correct session worker via `worker_url`
- Persisting checkpoint records returned by `/save`
- Updating `last_active_at` after each proxied command
- Idle-timeout enforcement (Scheduler)
- DB as source of truth for job/session status

### Job runner owns

- Training execution (real or simulated)
- Incremental progress and log reporting
- Terminal status reporting (`/complete` or `/fail`)
- Console callback notifications
- Usage event emission to the metering endpoint

### Session worker owns

- Model lifecycle in accelerator memory
- HTTP server for command dispatch
- Readiness registration with the server
- Checkpoint writes to shared storage
- Graceful shutdown on `/shutdown`
