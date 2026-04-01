"""
Celery task: Dataset Analysis Worker.
Idempotent — safe to re-run for same job_id.
"""
import logging
import traceback
from datetime import datetime, timezone

from celery import shared_task

from app.db.session import get_sync_db
from app.models.dataset import Dataset, Job, DatasetProfile
from app.services.analysis import run_full_analysis

logger = logging.getLogger(__name__)


@shared_task(
    name="app.workers.analyze_task.analyze_dataset",
    bind=True,
    max_retries=3,
    acks_late=True,
    default_retry_delay=5,
)
def analyze_dataset(self, dataset_id: str, job_id: str):
    """
    Analyze a dataset: stream from S3 → profile → LLM suggestions → write to DB.
    Idempotent: checks if job is already completed before running.
    """
    db = get_sync_db()
    try:
        # Check idempotency — skip if already completed
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            logger.error("Job not found", extra={"job_id": job_id})
            return {"status": "error", "message": "Job not found"}

        if job.status in ("completed", "degraded"):
            logger.info("Job already completed, skipping", extra={"job_id": job_id})
            return {"status": job.status, "message": "Already completed"}

        # Mark as processing
        job.status = "processing"
        job.started_at = datetime.now(timezone.utc)
        db.commit()

        # Get dataset info
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            job.status = "failed"
            job.error_message = "Dataset not found"
            job.completed_at = datetime.now(timezone.utc)
            db.commit()
            return {"status": "failed", "message": "Dataset not found"}

        # Run analysis
        result = run_full_analysis(s3_key=dataset.s3_raw_key)

        # Save profile
        profile = DatasetProfile(
            dataset_id=dataset_id,
            row_count=result.get("row_count"),
            column_count=result.get("column_count"),
            schema_snapshot=result.get("schema_snapshot"),
            quality_score=result.get("quality_score"),
            null_stats=result.get("null_stats"),
            duplicate_count=result.get("duplicate_count"),
            llm_suggestions=result.get("llm_suggestions", []),
        )
        db.add(profile)

        # Update job status
        is_degraded = not result.get("llm_assisted", True)
        job.status = "degraded" if is_degraded else "completed"
        job.completed_at = datetime.now(timezone.utc)
        job.metadata_json = {
            "tier_used": result.get("tier_used"),
            "warnings": result.get("warnings", []),
            "llm_assisted": result.get("llm_assisted"),
        }

        # Update dataset status
        dataset.status = "analyzed"
        db.commit()

        logger.info(
            "Analysis task completed",
            extra={"job_id": job_id, "dataset_id": dataset_id, "status": job.status},
        )
        return {"status": job.status, "profile_id": str(profile.id)}

    except Exception as exc:
        db.rollback()
        logger.error(
            "Analysis task failed",
            extra={"job_id": job_id, "error": str(exc), "traceback": traceback.format_exc()},
        )

        # Update job with error info
        try:
            job = db.query(Job).filter(Job.id == job_id).first()
            if job:
                job.retry_count = (job.retry_count or 0) + 1
                job.metadata_json = {
                    **(job.metadata_json or {}),
                    "last_error": str(exc),
                    "last_traceback": traceback.format_exc()[-500:],
                }
                if self.request.retries >= self.max_retries:
                    job.status = "failed"
                    job.error_message = str(exc)
                    job.completed_at = datetime.now(timezone.utc)
                db.commit()
        except Exception:
            pass

        raise self.retry(exc=exc, countdown=2 ** self.request.retries * 5)

    finally:
        db.close()
