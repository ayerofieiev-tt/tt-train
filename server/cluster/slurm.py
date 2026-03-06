from __future__ import annotations

import asyncio
import os
from pathlib import Path

from server.cluster.base import ClusterBackend, BackendState

LOG_DIR = "/tmp/tt_train_logs"

# Map Slurm job state codes to our BackendState
_SLURM_STATE_MAP: dict[str, BackendState] = {
    "PENDING": "pending",
    "CONFIGURING": "pending",
    "RESIZING": "pending",
    "SUSPENDED": "pending",
    "RUNNING": "running",
    "COMPLETING": "running",
    "COMPLETED": "done",
    "FAILED": "failed",
    "TIMEOUT": "failed",
    "NODE_FAIL": "failed",
    "PREEMPTED": "failed",
    "BOOT_FAIL": "failed",
    "DEADLINE": "failed",
    "OUT_OF_MEMORY": "failed",
    "CANCELLED": "cancelled",
    "CANCELLED+": "cancelled",
    "REVOKED": "cancelled",
}


class SlurmBackend(ClusterBackend):
    """Cluster backend that submits jobs via Slurm's sbatch/scancel/squeue."""

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
        return await self._sbatch(
            job_name=f"tt-job-{job_id}",
            stdout=f"{LOG_DIR}/job_{job_id}.out",
            stderr=f"{LOG_DIR}/job_{job_id}.err",
            nodes=nodes,
            partition=partition,
            account=account,
            script_path=script_path,
            script_args=args,
            env=env,
        )

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
        return await self._sbatch(
            job_name=f"tt-sess-{session_id}",
            stdout=f"{LOG_DIR}/sess_{session_id}.out",
            stderr=f"{LOG_DIR}/sess_{session_id}.err",
            nodes=nodes,
            partition=partition,
            account=account,
            script_path=script_path,
            script_args=args,
            env=env,
        )

    async def cancel(self, backend_job_id: str) -> None:
        returncode, _stdout, stderr = await self._run(["scancel", backend_job_id])
        if returncode != 0:
            # scancel can return non-zero if the job is already done; log but don't raise.
            # Only raise if the error is clearly unexpected.
            if "Invalid job id" not in stderr and "error" in stderr.lower():
                raise RuntimeError(f"scancel failed for job {backend_job_id}: {stderr}")

    async def get_state(self, backend_job_id: str) -> BackendState:
        # -h suppresses the header; %T outputs the job state string
        returncode, stdout, _stderr = await self._run(
            ["squeue", "-j", backend_job_id, "-h", "-o", "%T"]
        )
        if returncode != 0 or not stdout:
            # squeue returns non-zero / empty when the job is no longer in the queue,
            # which typically means it completed (or was never submitted to this cluster).
            return "done"

        # squeue may return multiple lines for array jobs; take the first.
        raw_state = stdout.splitlines()[0].strip().upper()
        # Strip trailing '+' that Slurm appends to some states (e.g. "CANCELLED+")
        return _SLURM_STATE_MAP.get(raw_state, _SLURM_STATE_MAP.get(raw_state.rstrip("+"), "done"))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _write_sbatch_script(
        self,
        *,
        job_name: str,
        stdout: str,
        stderr: str,
        nodes: int,
        partition: str | None,
        account: str | None,
        script_path: str,
        args: list[str],
    ) -> str:
        """Write a temporary sbatch shell script and return its path."""
        import tempfile
        import shlex
        from server.config import settings

        directives = [
            "#!/bin/bash",
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --nodes={nodes}",
            f"#SBATCH --output={stdout}",
            f"#SBATCH --error={stderr}",
        ]
        if partition:
            directives.append(f"#SBATCH --partition={partition}")
        if account:
            directives.append(f"#SBATCH --account={account}")

        body_lines = []
        if settings.slurm_venv_path:
            body_lines.append(f"source {shlex.quote(settings.slurm_venv_path)}/bin/activate")

        # Always inject repo root into PYTHONPATH
        repo_root = str(Path(__file__).parent.parent.parent)
        body_lines.append(f'export PYTHONPATH="{repo_root}:${{PYTHONPATH}}"')

        # Build the python command
        cmd_parts = ["python", shlex.quote(script_path)] + [shlex.quote(a) for a in args]
        body_lines.append(" ".join(cmd_parts))

        script_content = "\n".join(directives + [""] + body_lines) + "\n"

        os.makedirs(settings.slurm_script_tmpdir, exist_ok=True)
        fd, script_file = tempfile.mkstemp(
            suffix=".sh", prefix="tt_", dir=settings.slurm_script_tmpdir
        )
        with os.fdopen(fd, "w") as f:
            f.write(script_content)
        os.chmod(script_file, 0o755)
        return script_file

    async def _sbatch(
        self,
        *,
        job_name: str,
        stdout: str,
        stderr: str,
        nodes: int,
        partition: str | None,
        account: str | None,
        script_path: str,
        script_args: list[str],
        env: dict[str, str] | None,
    ) -> str:
        """Generate sbatch script, submit it, clean up, return Slurm job ID."""
        tmp_script = self._write_sbatch_script(
            job_name=job_name,
            stdout=stdout,
            stderr=stderr,
            nodes=nodes,
            partition=partition,
            account=account,
            script_path=script_path,
            args=script_args,
        )
        try:
            returncode, stdout_out, stderr_out = await self._run(["sbatch", tmp_script], env)
        finally:
            try:
                os.unlink(tmp_script)
            except OSError:
                pass

        if returncode != 0:
            raise RuntimeError(f"sbatch failed (exit {returncode}): {stderr_out}")
        for token in reversed(stdout_out.split()):
            if token.isdigit():
                return token
        raise RuntimeError(f"Could not parse sbatch output: {stdout_out!r}")

    async def _run(
        self, cmd: list[str], env: dict[str, str] | None = None
    ) -> tuple[int, str, str]:
        """Run a subprocess, return (returncode, stdout, stderr)."""
        proc_env = os.environ.copy()
        if env:
            proc_env.update(env)
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=proc_env,
        )
        stdout, stderr = await proc.communicate()
        return proc.returncode, stdout.decode().strip(), stderr.decode().strip()
