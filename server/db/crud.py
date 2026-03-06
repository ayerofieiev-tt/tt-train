from __future__ import annotations

from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession

from server.db.models import Checkpoint, Dataset, Job, RewardFunction, Session, TrainingLog, TrainingMetric


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------


async def create_job(
    db: AsyncSession,
    *,
    id: str,
    model: str,
    method: str,
    training_data: str,
    **kwargs,
) -> Job:
    job = Job(id=id, model=model, method=method, training_data=training_data, **kwargs)
    db.add(job)
    await db.commit()
    await db.refresh(job)
    return job


async def get_job(db: AsyncSession, job_id: str) -> Job | None:
    result = await db.execute(select(Job).where(Job.id == job_id))
    return result.scalar_one_or_none()


async def list_jobs(
    db: AsyncSession,
    *,
    limit: int = 20,
    after: str | None = None,
    status: str | None = None,
    method: str | None = None,
) -> list[Job]:
    stmt = select(Job).order_by(Job.created_at.desc(), Job.id.desc())
    if after is not None:
        stmt = stmt.where(Job.id < after)
    if status is not None:
        stmt = stmt.where(Job.status == status)
    if method is not None:
        stmt = stmt.where(Job.method == method)
    stmt = stmt.limit(limit)
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def update_job(db: AsyncSession, job_id: str, **kwargs) -> Job | None:
    if not kwargs:
        return await get_job(db, job_id)
    await db.execute(update(Job).where(Job.id == job_id).values(**kwargs))
    await db.commit()
    return await get_job(db, job_id)


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


async def create_session(
    db: AsyncSession,
    *,
    id: str,
    model: str,
    **kwargs,
) -> Session:
    session = Session(id=id, model=model, **kwargs)
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return session


async def get_session(db: AsyncSession, session_id: str) -> Session | None:
    result = await db.execute(select(Session).where(Session.id == session_id))
    return result.scalar_one_or_none()


async def list_sessions(
    db: AsyncSession,
    *,
    limit: int = 20,
    after: str | None = None,
    status: str | None = None,
) -> list[Session]:
    stmt = select(Session).order_by(Session.created_at.desc(), Session.id.desc())
    if after is not None:
        stmt = stmt.where(Session.id < after)
    if status is not None:
        stmt = stmt.where(Session.status == status)
    stmt = stmt.limit(limit)
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def update_session(db: AsyncSession, session_id: str, **kwargs) -> Session | None:
    if not kwargs:
        return await get_session(db, session_id)
    await db.execute(update(Session).where(Session.id == session_id).values(**kwargs))
    await db.commit()
    return await get_session(db, session_id)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


async def create_dataset(
    db: AsyncSession,
    *,
    id: str,
    format: str,
    **kwargs,
) -> Dataset:
    dataset = Dataset(id=id, format=format, **kwargs)
    db.add(dataset)
    await db.commit()
    await db.refresh(dataset)
    return dataset


async def get_dataset(db: AsyncSession, dataset_id: str) -> Dataset | None:
    result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    return result.scalar_one_or_none()


async def list_datasets(
    db: AsyncSession,
    *,
    limit: int = 20,
    after: str | None = None,
    format: str | None = None,
) -> list[Dataset]:
    stmt = select(Dataset).order_by(Dataset.created_at.desc(), Dataset.id.desc())
    if after is not None:
        stmt = stmt.where(Dataset.id < after)
    if format is not None:
        stmt = stmt.where(Dataset.format == format)
    stmt = stmt.limit(limit)
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def delete_dataset(db: AsyncSession, dataset_id: str) -> bool:
    result = await db.execute(delete(Dataset).where(Dataset.id == dataset_id))
    await db.commit()
    return result.rowcount > 0


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


async def create_checkpoint(
    db: AsyncSession,
    *,
    id: str,
    model_path: str,
    step: int,
    **kwargs,
) -> Checkpoint:
    checkpoint = Checkpoint(id=id, model_path=model_path, step=step, **kwargs)
    db.add(checkpoint)
    await db.commit()
    await db.refresh(checkpoint)
    return checkpoint


async def list_checkpoints_for_job(db: AsyncSession, job_id: str) -> list[Checkpoint]:
    result = await db.execute(
        select(Checkpoint)
        .where(Checkpoint.job_id == job_id)
        .order_by(Checkpoint.created_at.asc())
    )
    return list(result.scalars().all())


async def list_checkpoints_for_session(db: AsyncSession, session_id: str) -> list[Checkpoint]:
    result = await db.execute(
        select(Checkpoint)
        .where(Checkpoint.session_id == session_id)
        .order_by(Checkpoint.created_at.asc())
    )
    return list(result.scalars().all())


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------


async def create_reward(db: AsyncSession, *, id: str, **kwargs) -> RewardFunction:
    reward = RewardFunction(id=id, **kwargs)
    db.add(reward)
    await db.commit()
    await db.refresh(reward)
    return reward


async def get_reward(db: AsyncSession, reward_id: str) -> RewardFunction | None:
    result = await db.execute(select(RewardFunction).where(RewardFunction.id == reward_id))
    return result.scalar_one_or_none()


async def list_rewards(
    db: AsyncSession,
    *,
    limit: int = 20,
    after: str | None = None,
) -> list[RewardFunction]:
    stmt = select(RewardFunction).order_by(RewardFunction.created_at.desc(), RewardFunction.id.desc())
    if after is not None:
        stmt = stmt.where(RewardFunction.id < after)
    stmt = stmt.limit(limit)
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def delete_reward(db: AsyncSession, reward_id: str) -> bool:
    result = await db.execute(delete(RewardFunction).where(RewardFunction.id == reward_id))
    await db.commit()
    return result.rowcount > 0


# ---------------------------------------------------------------------------
# Training Metrics
# ---------------------------------------------------------------------------


async def record_metric(
    db: AsyncSession,
    *,
    job_id: str,
    step: int,
    **kwargs,
) -> None:
    try:
        metric = TrainingMetric(job_id=job_id, step=step, **kwargs)
        db.add(metric)
        await db.commit()
    except Exception:
        await db.rollback()


async def list_metrics_for_job(db: AsyncSession, job_id: str) -> list[TrainingMetric]:
    result = await db.execute(
        select(TrainingMetric)
        .where(TrainingMetric.job_id == job_id)
        .order_by(TrainingMetric.step.asc())
    )
    return list(result.scalars().all())


# ---------------------------------------------------------------------------
# Training Logs
# ---------------------------------------------------------------------------


async def append_log(
    db: AsyncSession,
    *,
    job_id: str,
    log_type: str = "info",
    message: str,
    step: int | None = None,
) -> TrainingLog:
    import uuid as _uuid
    entry = TrainingLog(
        id=str(_uuid.uuid4()),
        job_id=job_id,
        step=step,
        log_type=log_type,
        message=message,
    )
    db.add(entry)
    await db.commit()
    await db.refresh(entry)
    return entry


async def list_logs_for_job(
    db: AsyncSession,
    job_id: str,
    *,
    log_type: str | None = None,
    limit: int = 500,
) -> list[TrainingLog]:
    stmt = (
        select(TrainingLog)
        .where(TrainingLog.job_id == job_id)
        .order_by(TrainingLog.logged_at.asc())
        .limit(limit)
    )
    if log_type is not None:
        stmt = stmt.where(TrainingLog.log_type == log_type)
    result = await db.execute(stmt)
    return list(result.scalars().all())
