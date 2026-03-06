from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from server.cluster.base import ClusterBackend, BackendState

LOG_DIR = "/tmp/tt_train_logs"


class LocalBackend(ClusterBackend):
    """Cluster backend that runs scripts as local Python subprocesses.

    Intended for development without a real Slurm cluster.  Process state is
    tracked entirely in memory via a dict of asyncio ``Process`` objects.
    """

    def __init__(self) -> None:
        from server.config import settings
        self._procs: dict[str, asyncio.subprocess.Process] = {}
        self._counter: int = 0
        self.max_concurrent_jobs: int | None = settings.local_max_concurrent_jobs
        self.max_concurrent_sessions: int | None = settings.local_max_concurrent_sessions

    def _merged_env(self, extra: dict[str, str] | None) -> dict[str, str]:
        """Build subprocess env: inherit current env, inject PYTHONPATH so
        ``from workers.common import ...`` resolves when the script is run
        directly (the tt-train repo root must be importable)."""
        env = {**os.environ, **(extra or {})}
        repo_root = str(Path(__file__).parent.parent.parent)  # .../tt-train
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{repo_root}:{existing}" if existing else repo_root
        return env

    async def submit_job(
        self,
        *,
        job_id: str,
        script_path: str,
        args: list[str],
        nodes: int = 1,
        partition: str | None = None,
        account: str | None = None,
        env: dict[str, str] | None = None,
    ) -> str:
        os.makedirs(LOG_DIR, exist_ok=True)
        backend_id = f"local-{self._counter}"
        self._counter += 1
        merged_env = self._merged_env(env)
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            script_path,
            *args,
            stdout=open(f"{LOG_DIR}/job_{job_id}.out", "w"),
            stderr=open(f"{LOG_DIR}/job_{job_id}.err", "w"),
            env=merged_env,
        )
        self._procs[backend_id] = proc
        return backend_id

    async def submit_session(
        self,
        *,
        session_id: str,
        script_path: str,
        args: list[str],
        nodes: int = 1,
        partition: str | None = None,
        account: str | None = None,
        env: dict[str, str] | None = None,
    ) -> str:
        os.makedirs(LOG_DIR, exist_ok=True)
        backend_id = f"local-sess-{self._counter}"
        self._counter += 1
        merged_env = self._merged_env(env)
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            script_path,
            *args,
            stdout=open(f"{LOG_DIR}/sess_{session_id}.out", "w"),
            stderr=open(f"{LOG_DIR}/sess_{session_id}.err", "w"),
            env=merged_env,
        )
        self._procs[backend_id] = proc
        return backend_id

    async def cancel(self, backend_job_id: str) -> None:
        proc = self._procs.get(backend_job_id)
        if proc is not None and proc.returncode is None:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()

    async def get_state(self, backend_job_id: str) -> BackendState:
        proc = self._procs.get(backend_job_id)
        if proc is None:
            return "done"
        if proc.returncode is None:
            # Process is still alive — check via /proc to be safe.
            if not _proc_is_alive(proc.pid):
                # Process died without us noticing; reap it.
                try:
                    await asyncio.wait_for(proc.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    pass
                return "failed" if (proc.returncode or 0) != 0 else "done"
            return "running"
        if proc.returncode == 0:
            return "done"
        # Negative returncode means the process was killed by a signal (e.g. SIGTERM from cancel()).
        return "cancelled" if proc.returncode < 0 else "failed"


def _proc_is_alive(pid: int) -> bool:
    """Return True if *pid* refers to a live process (Linux-specific /proc check)."""
    return os.path.exists(f"/proc/{pid}/status")
