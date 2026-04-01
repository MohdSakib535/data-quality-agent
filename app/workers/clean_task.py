"""
Celery task: Dataset Cleaning Worker.
Idempotent — safe to re-run for same job_id.
"""
import json
import logging
import traceback
from datetime import datetime, timezone

from celery import shared_task

from app.db.session import get_sync_db
from app.models.dataset import Dataset, Job, DatasetProfile, CleanedDataPreview
from app.models.query import ParquetSchemaVersion
from app.services.cleaning import run_cleaning_pipeline
from app.utils.hashing import schema_fingerprint
from app.core.config import settings

logger = logging.getLogger(__name__)


@shared_task(
    name="app.workers.clean_task.clean_dataset",
    bind=True,
    max_retries=3,
    acks_late=True,
    default_retry_delay=10,
)
def clean_dataset(self, dataset_id: str, job_id: str, prompt: str):
    """
    Clean a dataset: deterministic + LLM-assisted cleaning → save Parquet → preview.
    Idempotent: checks if job is already completed before running.
    """
    db = get_sync_db()
    try:
        # Check idempotency
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

        # Get dataset
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            job.status = "failed"
            job.error_message = "Dataset not found"
            job.completed_at = datetime.now(timezone.utc)
            db.commit()
            return {"status": "failed", "message": "Dataset not found"}

        # Get detected issues from profile (if any)
        profile = db.query(DatasetProfile).filter(
            DatasetProfile.dataset_id == dataset_id
        ).order_by(DatasetProfile.profiled_at.desc()).first()

        detected_issues = []
        if profile and profile.llm_suggestions:
            detected_issues = profile.llm_suggestions

        # Run cleaning pipeline
        result = run_cleaning_pipeline(
            s3_raw_key=dataset.s3_raw_key,
            user_id=str(dataset.user_id),
            dataset_id=str(dataset_id),
            job_id=str(job_id),
            prompt=prompt,
            detected_issues=detected_issues,
        )

        # Update dataset with cleaned key
        dataset.s3_cleaned_key = result["s3_parquet_key"]
        dataset.status = "cleaned"

        # Save parquet schema version
        existing_version = db.query(ParquetSchemaVersion).filter(
            ParquetSchemaVersion.dataset_id == dataset_id
        ).order_by(ParquetSchemaVersion.version.desc()).first()

        new_version = 1
        if existing_version:
            if existing_version.schema_fingerprint != result["schema_fingerprint"]:
                new_version = existing_version.version + 1
            else:
                new_version = existing_version.version

        # Only create new version if fingerprint changed or no version exists
        if not existing_version or existing_version.schema_fingerprint != result["schema_fingerprint"]:
            schema_version = ParquetSchemaVersion(
                dataset_id=dataset_id,
                version=new_version,
                schema_fingerprint=result["schema_fingerprint"],
                column_map=result["original_column_map"],
                parquet_metadata={
                    "s3_key": result["s3_parquet_key"],
                    "row_count": result["cleaned_row_count"],
                    "llm_assisted": result["llm_assisted"],
                },
            )
            db.add(schema_version)

        # Save preview
        preview = CleanedDataPreview(
            dataset_id=dataset_id,
            job_id=job_id,
            preview_rows=result["preview_rows"],
            cleaned_row_count=result["cleaned_row_count"],
            prompt_used=prompt,
            cleaning_steps_applied=result["cleaning_steps_applied"],
        )
        db.add(preview)

        # Update job status
        is_degraded = result.get("degraded_mode", False)
        job.status = "degraded" if is_degraded else "completed"
        job.completed_at = datetime.now(timezone.utc)
        job.metadata_json = {
            "cleaned_row_count": result["cleaned_row_count"],
            "steps_count": len(result["cleaning_steps_applied"]),
            "llm_assisted": result["llm_assisted"],
            "warnings": result.get("warnings", []),
            "assumptions": result.get("assumptions", []),
        }

        db.commit()

        logger.info(
            "Cleaning task completed",
            extra={
                "job_id": job_id,
                "dataset_id": dataset_id,
                "status": job.status,
                "rows": result["cleaned_row_count"],
            },
        )
        return {"status": job.status, "cleaned_rows": result["cleaned_row_count"]}

    except Exception as exc:
        db.rollback()
        logger.error(
            "Cleaning task failed",
            extra={"job_id": job_id, "error": str(exc), "traceback": traceback.format_exc()},
        )

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

        raise self.retry(exc=exc, countdown=2 ** self.request.retries * 10)

    finally:
        db.close()
