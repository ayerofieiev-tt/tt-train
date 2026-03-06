from __future__ import annotations

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.sql import func

from server.db.engine import Base


class Job(Base):
    __tablename__ = "jobs"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=True)
    model = Column(String, nullable=False)
    method = Column(String, nullable=False)  # sft/dpo/rl/pretrain
    status = Column(String, nullable=False, default="queued")  # queued/running/paused/completed/failed/cancelled
    training_data = Column(String, nullable=False)
    validation_data = Column(String, nullable=True)
    console_job_id = Column(String, nullable=True, index=True)
    dataset_url = Column(String, nullable=True)
    console_base_url = Column(String, nullable=True)
    worker_token = Column(String, nullable=True)
    callback_url = Column(String, nullable=True)
    config = Column(JSON, default={})
    hardware_config = Column(JSON, default={})  # renamed from 'hardware' to avoid ORM conflicts
    slurm_job_id = Column(String, nullable=True)  # the backend job id
    progress = Column(JSON, nullable=True)
    metrics = Column(JSON, nullable=True)
    result_model = Column(String, nullable=True)
    error = Column(JSON, nullable=True)
    cost = Column(JSON, nullable=True)
    webhooks = Column(JSON, nullable=True)
    metadata_ = Column("metadata", JSON, default={})  # renamed from 'metadata' to avoid Python keyword
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)


class Session(Base):
    __tablename__ = "sessions"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=True)
    model = Column(String, nullable=False)
    status = Column(String, nullable=False, default="provisioning")  # provisioning/ready/closed/expired/failed
    lora_config = Column(JSON, nullable=True)
    optimizer_config = Column(JSON, nullable=True)
    hardware_config = Column(JSON, default={})
    slurm_job_id = Column(String, nullable=True)
    worker_url = Column(String, nullable=True)  # http://nodeXXX:PORT — how API server reaches worker
    step_count = Column(Integer, default=0)
    total_cost = Column(String, default="$0.00")
    idle_timeout_minutes = Column(Integer, default=30)
    last_checkpoint = Column(String, nullable=True)
    last_active_at = Column(DateTime(timezone=True), nullable=True)
    metadata_ = Column("metadata", JSON, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=True)
    closed_at = Column(DateTime(timezone=True), nullable=True)


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=True)
    description = Column(String, nullable=True)
    format = Column(String, nullable=False)
    status = Column(String, nullable=False, default="processing")  # processing/ready/failed/deleted
    bytes = Column(Integer, nullable=True)
    storage_path = Column(String, nullable=True)
    stats = Column(JSON, nullable=True)
    metadata_ = Column("metadata", JSON, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Checkpoint(Base):
    __tablename__ = "checkpoints"

    id = Column(String, primary_key=True)
    model_path = Column(String, nullable=False)
    job_id = Column(String, ForeignKey("jobs.id"), nullable=True)
    session_id = Column(String, ForeignKey("sessions.id"), nullable=True)
    step = Column(Integer, nullable=False)
    epoch = Column(Float, nullable=True)
    name = Column(String, nullable=True)
    metrics = Column(JSON, default={})
    metadata_ = Column("metadata", JSON, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class TrainingMetric(Base):
    __tablename__ = "training_metrics"
    __table_args__ = (UniqueConstraint("job_id", "step"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String, ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    step = Column(Integer, nullable=False)
    epoch = Column(Float, nullable=True)
    train_loss = Column(Float, nullable=True)
    val_loss = Column(Float, nullable=True)
    grad_norm = Column(Float, nullable=True)
    learning_rate = Column(Float, nullable=True)
    tokens_per_second = Column(Float, nullable=True)
    recorded_at = Column(DateTime(timezone=True), server_default=func.now())


class TrainingLog(Base):
    __tablename__ = "training_logs"

    id = Column(String, primary_key=True)
    job_id = Column(String, ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    step = Column(Integer, nullable=True)
    log_type = Column(String, nullable=False, default="info")
    message = Column(Text, nullable=False)
    logged_at = Column(DateTime(timezone=True), server_default=func.now())


class RewardFunction(Base):
    __tablename__ = "reward_functions"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=True)
    status = Column(String, nullable=False, default="validating")
    runtime = Column(String, default="python3.11")
    storage_path = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
