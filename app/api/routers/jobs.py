"""
Job status API router.
"""
import uuid
import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models.dataset import Job
from app.schemas.job import JobStatusResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/jobs", tags=["jobs"])


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """
    Get the status of a job.
    If status is 'degraded', includes a warning explaining which steps ran without LLM.
    """
    result = await db.execute(select(Job).where(Job.id == job_id))
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Build warning for degraded jobs
    warning = None
    if job.status == "degraded":
        meta = job.metadata_json or {}
        warnings = meta.get("warnings", [])
        if warnings:
            warning = "; ".join(warnings)
        else:
            warning = "Job completed with partial results — LLM fallback was triggered"

    return JobStatusResponse(
        id=job.id,
        dataset_id=job.dataset_id,
        job_type=job.job_type,
        status=job.status,
        started_at=job.started_at,
        completed_at=job.completed_at,
        error_message=job.error_message,
        retry_count=job.retry_count,
        metadata=job.metadata_json or {},
        warning=warning,
    )
