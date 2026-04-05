import logging
import os
import tempfile

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.background import BackgroundTask

from app.core.config import settings
from app.db.session import get_db
from app.models.cleaned_data import CleanedData
from app.models.job import Job
from app.schemas.job import CleanDataRequest, CleanDataResponse
from app.services.chat_service import (
    get_cleaned_data_row_samples,
    get_cleaned_data_row_schema,
    get_table_schema,
)
from app.services.object_storage import cleaned_output_key, get_object_storage_service
from app.services.pipeline import clean_csv_with_prompt
from app.services.semantic_layer import ensure_semantic_metadata

router = APIRouter(prefix="/clean", tags=["clean"])
logger = logging.getLogger(__name__)


@router.post("/{job_id}", response_model=CleanDataResponse)
async def clean_job(
    job_id: str,
    request: CleanDataRequest,
    db: AsyncSession = Depends(get_db),
):
    """Run pure AI cleaning pipeline for a job based on a chosen prompt."""
    result = await db.execute(select(Job).filter(Job.id == job_id))
    db_job = result.scalars().first()

    if not db_job:
        raise HTTPException(status_code=404, detail="Job ID not found.")

    db_job.status = "processing"
    await db.commit()

    try:
        response = await clean_csv_with_prompt(job_id, request.prompt)
        cleaned_result = await db.get(CleanedData, job_id)

        if cleaned_result is None:
            cleaned_result = CleanedData(
                job_id=job_id,
                prompt=request.prompt,
                cleaned_file_path=get_object_storage_service().object_uri(cleaned_output_key(job_id)),
                cleaned_data=response.cleaned_data,
            )
            db.add(cleaned_result)
        else:
            cleaned_result.prompt = request.prompt
            cleaned_result.cleaned_file_path = get_object_storage_service().object_uri(cleaned_output_key(job_id))
            cleaned_result.cleaned_data = response.cleaned_data

        db_job.status = "completed"
        db_job.message = None
        await db.commit()

        try:
            table_schema = await get_table_schema("cleaned_data")
            row_schema = await get_cleaned_data_row_schema(job_id)
            row_samples = await get_cleaned_data_row_samples(
                job_id,
                sample_items=settings.SEMANTIC_ROW_SAMPLE_LIMIT,
            )
            await ensure_semantic_metadata(
                job_id=job_id,
                table_name="cleaned_data",
                table_schema=table_schema,
                row_schema=row_schema,
                row_samples=row_samples,
            )
        except Exception:
            logger.exception("Failed to build semantic metadata for job_id=%s", job_id)

        return response
    except Exception as exc:
        await db.rollback()
        db_job.status = "failed"
        db_job.message = str(exc)
        await db.commit()
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/{job_id}/download")
async def download_cleaned_file(job_id: str, db: AsyncSession = Depends(get_db)):
    """Download the final cleaned CSV file."""
    cleaned_result = await db.get(CleanedData, job_id)
    if cleaned_result is None:
        raise HTTPException(status_code=404, detail="Cleaned data not found. Run cleaning first.")

    fd, path = tempfile.mkstemp(prefix=f"{job_id}_cleaned_", suffix=".csv")
    os.close(fd)
    restored = get_object_storage_service().download_file(
        cleaned_output_key(job_id),
        path,
    )
    if not restored:
        if os.path.exists(path):
            os.remove(path)
        raise HTTPException(status_code=404, detail="Cleaned file not found. Run cleaning again.")

    return FileResponse(
        path,
        media_type="text/csv",
        filename=f"{job_id}_cleaned.csv",
        background=BackgroundTask(lambda: os.path.exists(path) and os.remove(path)),
    )
