"""
Celery task: Query Worker (optional async query execution).
For queries that might take too long, this can be used instead of inline execution.
"""
import logging
import traceback
from datetime import datetime, timezone

from celery import shared_task

from app.db.session import get_sync_db
from app.models.dataset import Dataset, Job
from app.models.query import QueryLog, ParquetSchemaVersion
from app.services.nl_to_sql import run_nl_to_sql_pipeline

logger = logging.getLogger(__name__)


@shared_task(
    name="app.workers.query_task.execute_query",
    bind=True,
    max_retries=2,
    acks_late=True,
    default_retry_delay=3,
    time_limit=60,
    soft_time_limit=45,
)
def execute_query(self, dataset_id: str, job_id: str, question: str, user_id: str):
    """
    Execute an NL→SQL query asynchronously.
    Used for potentially long-running queries.
    """
    db = get_sync_db()
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if not job:
            return {"status": "error", "message": "Job not found"}

        if job.status in ("completed", "degraded"):
            return {"status": job.status, "message": "Already completed"}

        job.status = "processing"
        job.started_at = datetime.now(timezone.utc)
        db.commit()

        # Get dataset and schema version
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset or not dataset.s3_cleaned_key:
            job.status = "failed"
            job.error_message = "Dataset not cleaned yet"
            job.completed_at = datetime.now(timezone.utc)
            db.commit()
            return {"status": "failed", "message": "Dataset not cleaned yet"}

        schema_version = db.query(ParquetSchemaVersion).filter(
            ParquetSchemaVersion.dataset_id == dataset_id
        ).order_by(ParquetSchemaVersion.version.desc()).first()

        if not schema_version:
            job.status = "failed"
            job.error_message = "No schema version found"
            job.completed_at = datetime.now(timezone.utc)
            db.commit()
            return {"status": "failed", "message": "No schema version found"}

        # Run NL→SQL pipeline
        result = run_nl_to_sql_pipeline(
            question=question,
            dataset_id=str(dataset_id),
            s3_parquet_key=dataset.s3_cleaned_key,
            column_map=schema_version.column_map or {},
            parquet_metadata=schema_version.parquet_metadata or {},
        )

        # Log query
        query_log = QueryLog(
            dataset_id=dataset_id,
            natural_language_query=question,
            generated_sql=result.get("sql"),
            validated=result.get("validated", False),
            confidence_score=result.get("confidence"),
            execution_time_ms=result.get("execution_time_ms"),
            result_row_count=result.get("row_count"),
            error_message=result.get("error"),
        )
        db.add(query_log)

        # Update job
        if result.get("error"):
            job.status = "failed"
            job.error_message = result["error"]
        else:
            job.status = "completed"

        job.completed_at = datetime.now(timezone.utc)
        job.metadata_json = {"result": result}
        db.commit()

        return {"status": job.status, "result": result}

    except Exception as exc:
        db.rollback()
        logger.error(
            "Query task failed",
            extra={"job_id": job_id, "error": str(exc), "traceback": traceback.format_exc()},
        )
        try:
            job = db.query(Job).filter(Job.id == job_id).first()
            if job:
                job.retry_count = (job.retry_count or 0) + 1
                if self.request.retries >= self.max_retries:
                    job.status = "failed"
                    job.error_message = str(exc)
                    job.completed_at = datetime.now(timezone.utc)
                db.commit()
        except Exception:
            pass
        raise self.retry(exc=exc, countdown=2 ** self.request.retries * 3)
    finally:
        db.close()
