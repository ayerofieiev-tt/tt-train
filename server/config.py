from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "sqlite+aiosqlite:///./tt_train.db"
    api_key: str = ""  # if set, only this key accepted; else any non-empty key works
    cluster_backend: str = "local"  # "slurm" or "local"
    slurm_partition: str | None = None
    slurm_account: str | None = None
    internal_api_key: str = "internal-secret"  # used by workers to call /internal/*
    worker_script_dir: str = "/home/boxx/tt-train/workers"
    shared_storage_path: str = "/tmp/tt_train_storage"
    api_base_url: str = "http://localhost:8000/v1"  # how workers call back to API server
    scheduler_poll_interval: float = 5.0
    session_idle_timeout_minutes: int = 30
    # LocalBackend concurrency limits (None = unlimited; Slurm always uses None)
    local_max_concurrent_jobs: int = 1
    local_max_concurrent_sessions: int = 1

    # Console integration (optional — only needed when console is connected)
    console_base_url: str = ""           # e.g. https://console.api — empty = no console
    console_token_validate_url: str = "" # e.g. https://console.api/v1/tokens/validate

    # Slurm deployment
    slurm_venv_path: str = ""            # path to venv to activate on compute nodes, e.g. /shared/venv
    slurm_script_tmpdir: str = "/tmp/tt_train_sbatch"  # where sbatch scripts are written

    class Config:
        env_prefix = "TT_TRAIN_"
        env_file = ".env"


settings = Settings()
