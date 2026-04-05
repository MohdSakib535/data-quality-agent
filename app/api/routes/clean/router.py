import logging
import os
import tempfile

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.background import BackgroundTask

from app.api.error_utils import raise_http_for_service_error
from app.core.config import settings
from app.db.session import get_db
from app.models.analysis_suggestion import AnalysisSuggestion
from app.models.cleaned_data import CleanedData
from app.models.job import Job
from app.schemas.job import (
    CleanDataRequest,
    CleanDataResponse,
    CleanJobDetailResponse,
    CleanedFileListResponse,
    CleanedFileResponse,
)
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


def _clean_download_url(job_id: str) -> str:
    return f"/api/v1/clean/{job_id}/download"


def _build_clean_response(
    *,
    job_id: str,
    status: str,
    message: str | None = None,
    cleaned_result: CleanedData | None = None,
) -> CleanDataResponse:
    return CleanDataResponse(
        job_id=job_id,
        source_file_id=cleaned_result.source_file_id if cleaned_result is not None else job_id,
        status=status,
        cleaned_file_url=_clean_download_url(job_id),
        message=message,
        cleaned_rows=cleaned_result.cleaned_rows if cleaned_result is not None else 0,
        cleaned_data=cleaned_result.cleaned_data if status == "completed" and cleaned_result is not None else [],
    )


def _build_clean_detail_response(
    job: Job,
    cleaned_result: CleanedData | None,
) -> CleanJobDetailResponse:
    return CleanJobDetailResponse(
        job_id=job.id,
        source_file_id=cleaned_result.source_file_id if cleaned_result is not None else job.id,
        status=job.status,
        prompt=cleaned_result.prompt if cleaned_result is not None else None,
        cleaned_file_url=_clean_download_url(job.id),
        cleaned_file_path=cleaned_result.cleaned_file_path if cleaned_result is not None else None,
        cleaned_rows=cleaned_result.cleaned_rows if cleaned_result is not None else 0,
        cleaned_data=cleaned_result.cleaned_data if cleaned_result is not None and job.status == "completed" else [],
        quality_score=cleaned_result.quality_score if cleaned_result is not None else None,
        analysis=cleaned_result.analysis if cleaned_result is not None else None,
        message=job.message,
        created_at=cleaned_result.created_at if cleaned_result is not None else job.created_at,
        updated_at=cleaned_result.updated_at if cleaned_result is not None else job.updated_at,
    )


@router.get("", response_model=CleanedFileListResponse)
async def list_cleaned_files(db: AsyncSession = Depends(get_db)):
    """List all cleaned output files."""
    result = await db.execute(
        select(CleanedData, Job)
        .join(Job, Job.id == CleanedData.source_file_id)
        .order_by(CleanedData.created_at.desc())
    )
    cleaned_files = [
        CleanedFileResponse(
            job_id=cleaned_result.job_id,
            source_file_id=cleaned_result.source_file_id,
            status=job.status,
            filename=job.filename,
            prompt=cleaned_result.prompt,
            cleaned_file_path=cleaned_result.cleaned_file_path,
            cleaned_rows=cleaned_result.cleaned_rows,
            quality_score=cleaned_result.quality_score,
            analysis=cleaned_result.analysis,
            message=job.message,
            created_at=cleaned_result.created_at,
            updated_at=cleaned_result.updated_at,
        )
        for cleaned_result, job in result.all()
    ]
    return CleanedFileListResponse(cleaned_files=cleaned_files)


@router.get("/{job_id}", response_model=CleanJobDetailResponse)
async def get_clean_job(job_id: str, db: AsyncSession = Depends(get_db)):
    """Get cleaning status and latest cleaned output details for a job."""
    db_job = await db.get(Job, job_id)
    if db_job is None:
        raise HTTPException(status_code=404, detail="Job ID not found.")

    cleaned_result = await db.get(CleanedData, job_id)
    return _build_clean_detail_response(db_job, cleaned_result)


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

    cleaned_result = await db.get(CleanedData, job_id)
    db_job.status = "processing"
    db_job.message = None
    use_cleaned_source = cleaned_result is not None

    await db.execute(
        delete(AnalysisSuggestion).where(
            AnalysisSuggestion.job_id == job_id,
            AnalysisSuggestion.source_type == "clean",
        )
    )
    await db.commit()

    try:
        response = await clean_csv_with_prompt(
            job_id,
            request.prompt,
            use_cleaned_source=use_cleaned_source,
        )

        cleaned_result = await db.get(CleanedData, job_id)
        if cleaned_result is None:
            cleaned_result = CleanedData(
                job_id=job_id,
                source_file_id=job_id,
                prompt=request.prompt,
                cleaned_file_path=get_object_storage_service().object_uri(cleaned_output_key(job_id)),
                cleaned_data=response.cleaned_data,
                cleaned_rows=response.cleaned_rows,
                quality_score=None,
                analysis=None,
            )
            db.add(cleaned_result)
        else:
            cleaned_result.source_file_id = job_id
            cleaned_result.prompt = request.prompt
            cleaned_result.cleaned_file_path = get_object_storage_service().object_uri(cleaned_output_key(job_id))
            cleaned_result.cleaned_data = response.cleaned_data
            cleaned_result.cleaned_rows = response.cleaned_rows
            cleaned_result.quality_score = None
            cleaned_result.analysis = None

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

        return _build_clean_response(
            job_id=job_id,
            status="completed",
            cleaned_result=cleaned_result,
        )
    except Exception as exc:
        await db.rollback()
        db_job.status = "failed"
        db_job.message = str(exc)
        await db.commit()
        raise_http_for_service_error(exc, operation="Data cleaning", logger=logger)


@router.get("/{job_id}/download")
async def download_cleaned_file(job_id: str, db: AsyncSession = Depends(get_db)):
    """Download the final cleaned CSV file."""
    db_job = await db.get(Job, job_id)
    if db_job is None:
        raise HTTPException(status_code=404, detail="Job ID not found.")
    if db_job.status != "completed":
        raise HTTPException(status_code=409, detail="Cleaning is not completed yet.")

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
