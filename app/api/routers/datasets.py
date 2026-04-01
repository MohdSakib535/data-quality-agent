"""
Dataset API router: upload, analyze, clean, download endpoints.
"""
import os
import uuid
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File, Form
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models.dataset import Dataset, Job
from app.schemas.dataset import (
    InitiateUploadRequest,
    InitiateUploadResponse,
    CompleteUploadResponse,
    AnalyzeResponse,
    CleanRequest,
    CleanResponse,
    DownloadResponse,
)
from app.services.storage import (
    generate_presigned_put_url,
    generate_presigned_get_url,
    head_object,
    delete_object,
    validate_file_magic_bytes,
    upload_fileobj,
)
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/datasets", tags=["datasets"])


# ── 0. Direct Upload (multipart) ────────────────────────────
@router.post("/upload", response_model=CompleteUploadResponse)
async def upload_dataset_direct(
    user_id: uuid.UUID = Form(...),
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """
    Upload a dataset file via multipart form-data directly to the backend.
    Skips presigned URL flow. Saves to S3, creates dataset + analysis job, enqueues analysis.
    """
    filename = file.filename or "upload"
    _, ext = os.path.splitext(filename)
    ext = ext.lower()

    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {settings.ALLOWED_EXTENSIONS}",
        )

    # Determine file size (may spool to disk for large uploads)
    try:
        file.file.seek(0, os.SEEK_END)
        file_size_bytes = file.file.tell()
        file.file.seek(0)
    except Exception:
        file_size_bytes = 0

    if file_size_bytes <= 0 or file_size_bytes > settings.MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File too large or size unknown. Maximum: {settings.MAX_FILE_SIZE_BYTES / (1024*1024):.0f} MB",
        )

    dataset_id = uuid.uuid4()
    analyze_job_id = uuid.uuid4()
    s3_raw_key = f"raw/{user_id}/{dataset_id}/source{ext}"

    try:
        # Upload to S3
        content_type = file.content_type or "application/octet-stream"
        upload_fileobj(s3_raw_key, file.file, content_type=content_type)
        logger.info("Direct upload stored to S3", extra={"dataset_id": str(dataset_id), "s3_key": s3_raw_key})

        # Persist dataset + job
        dataset = Dataset(
            id=dataset_id,
            user_id=user_id,
            filename=filename,
            file_size_bytes=file_size_bytes,
            s3_raw_key=s3_raw_key,
            status="uploaded",
        )
        db.add(dataset)

        analyze_job = Job(
            id=analyze_job_id,
            dataset_id=dataset_id,
            job_type="analyze",
            status="queued",
        )
        db.add(analyze_job)
        await db.commit()
        logger.info("Direct upload DB commit succeeded", extra={"dataset_id": str(dataset_id), "job_id": str(analyze_job_id)})

        # Enqueue analysis
        from app.workers.analyze_task import analyze_dataset
        analyze_dataset.delay(str(dataset_id), str(analyze_job_id))
        logger.info("Direct upload analysis enqueued", extra={"dataset_id": str(dataset_id), "job_id": str(analyze_job_id)})

    except Exception as exc:
        await db.rollback()
        logger.exception("Direct upload failed", extra={"dataset_id": str(dataset_id), "error": str(exc)})
        raise HTTPException(status_code=500, detail="Failed to process upload") from exc

    logger.info(
        "Direct upload completed, analysis enqueued",
        extra={"dataset_id": str(dataset_id), "analyze_job_id": str(analyze_job_id)},
    )

    return CompleteUploadResponse(
        dataset_id=dataset_id,
        status="uploaded",
        analyze_job_id=analyze_job_id,
    )


# ── 1. Initiate Upload ───────────────────────────────────────
@router.post("/initiate-upload", response_model=InitiateUploadResponse)
async def initiate_upload(
    body: InitiateUploadRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Initiate a dataset upload. Returns a presigned S3 URL for the client to PUT the file.
    Validates filename extension and file size.
    """
    # Validate extension
    _, ext = os.path.splitext(body.filename)
    if ext.lower() not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {settings.ALLOWED_EXTENSIONS}",
        )

    # Validate file size
    if body.file_size_bytes > settings.MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum: {settings.MAX_FILE_SIZE_BYTES / (1024*1024):.0f} MB",
        )

    # Generate IDs
    dataset_id = uuid.uuid4()
    job_id = uuid.uuid4()

    # S3 key
    s3_raw_key = f"raw/{body.user_id}/{dataset_id}/source{ext.lower()}"

    # Generate presigned PUT URL
    upload_url, expires_at = generate_presigned_put_url(s3_raw_key)

    # Insert dataset row
    dataset = Dataset(
        id=dataset_id,
        user_id=body.user_id,
        filename=body.filename,
        file_size_bytes=body.file_size_bytes,
        s3_raw_key=s3_raw_key,
        status="pending_upload",
    )
    db.add(dataset)

    # Insert job row
    job = Job(
        id=job_id,
        dataset_id=dataset_id,
        job_type="analyze",
        status="queued",
    )
    db.add(job)

    await db.commit()

    logger.info(
        "Upload initiated",
        extra={"dataset_id": str(dataset_id), "filename": body.filename},
    )

    return InitiateUploadResponse(
        dataset_id=dataset_id,
        job_id=job_id,
        upload_url=upload_url,
        expires_at=expires_at,
    )


# ── 2. Complete Upload ───────────────────────────────────────
@router.post("/{dataset_id}/complete-upload", response_model=CompleteUploadResponse)
async def complete_upload(
    dataset_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """
    Confirm upload completion. Verifies S3 object exists, size matches, and magic bytes are valid.
    On success, enqueues analysis task.
    """
    # Get dataset
    result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if dataset.status != "pending_upload":
        raise HTTPException(status_code=400, detail=f"Dataset status is '{dataset.status}', expected 'pending_upload'")

    # Verify S3 object exists
    s3_meta = head_object(dataset.s3_raw_key)
    if not s3_meta:
        dataset.status = "upload_failed"
        await db.commit()
        raise HTTPException(status_code=400, detail="File not found in storage. Upload may have failed.")

    # Verify file size (±5% tolerance)
    actual_size = s3_meta["content_length"]
    declared_size = dataset.file_size_bytes
    tolerance = declared_size * settings.FILE_SIZE_TOLERANCE
    if abs(actual_size - declared_size) > tolerance:
        delete_object(dataset.s3_raw_key)
        dataset.status = "upload_failed"
        await db.commit()
        raise HTTPException(
            status_code=400,
            detail=f"File size mismatch. Declared: {declared_size}, Actual: {actual_size}",
        )

    # Validate magic bytes
    file_type = validate_file_magic_bytes(dataset.s3_raw_key)
    if file_type is None:
        delete_object(dataset.s3_raw_key)
        dataset.status = "upload_failed"
        await db.commit()
        raise HTTPException(status_code=400, detail="Invalid file format. Could not verify file type.")

    # Success — update status and enqueue analysis
    dataset.status = "uploaded"

    # Create analysis job
    analyze_job_id = uuid.uuid4()
    analyze_job = Job(
        id=analyze_job_id,
        dataset_id=dataset_id,
        job_type="analyze",
        status="queued",
    )
    db.add(analyze_job)
    await db.commit()

    # Enqueue analysis task (import here to avoid circular imports)
    from app.workers.analyze_task import analyze_dataset
    analyze_dataset.delay(str(dataset_id), str(analyze_job_id))

    logger.info(
        "Upload completed, analysis enqueued",
        extra={"dataset_id": str(dataset_id), "analyze_job_id": str(analyze_job_id)},
    )

    return CompleteUploadResponse(
        dataset_id=dataset_id,
        status="uploaded",
        analyze_job_id=analyze_job_id,
    )


# ── 3. Analyze Dataset ──────────────────────────────────────
@router.post("/{dataset_id}/analyze", response_model=AnalyzeResponse)
async def analyze_dataset_endpoint(
    dataset_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """
    Trigger dataset analysis. Idempotent: returns existing job if already running/completed.
    """
    # Check dataset exists
    result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Check for existing analysis job
    existing = await db.execute(
        select(Job).where(
            Job.dataset_id == dataset_id,
            Job.job_type == "analyze",
            Job.status.in_(["queued", "processing", "completed", "degraded"]),
        ).order_by(Job.created_at.desc())
    )
    existing_job = existing.scalar_one_or_none()

    if existing_job:
        return AnalyzeResponse(job_id=existing_job.id, status=existing_job.status)

    # Create new job and enqueue
    job_id = uuid.uuid4()
    job = Job(
        id=job_id,
        dataset_id=dataset_id,
        job_type="analyze",
        status="queued",
    )
    db.add(job)
    await db.commit()

    from app.workers.analyze_task import analyze_dataset
    analyze_dataset.delay(str(dataset_id), str(job_id))

    return AnalyzeResponse(job_id=job_id, status="queued")


# ── 4. Clean Dataset ────────────────────────────────────────
@router.post("/{dataset_id}/clean", response_model=CleanResponse)
async def clean_dataset_endpoint(
    dataset_id: uuid.UUID,
    body: CleanRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Trigger dataset cleaning with user-provided prompt.
    """
    # Check dataset exists
    result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Create new cleaning job
    job_id = uuid.uuid4()
    job = Job(
        id=job_id,
        dataset_id=dataset_id,
        job_type="clean",
        status="queued",
    )
    db.add(job)
    await db.commit()

    from app.workers.clean_task import clean_dataset
    clean_dataset.delay(str(dataset_id), str(job_id), body.prompt)

    logger.info(
        "Cleaning job enqueued",
        extra={"dataset_id": str(dataset_id), "job_id": str(job_id)},
    )

    return CleanResponse(job_id=job_id, status="queued")


# ── 6. Download Cleaned File ────────────────────────────────
@router.get("/{dataset_id}/download", response_model=DownloadResponse)
async def download_dataset(
    dataset_id: uuid.UUID,
    format: str = Query(default="parquet", pattern="^(csv|parquet)$"),
    db: AsyncSession = Depends(get_db),
):
    """
    Generate a presigned download URL for the cleaned dataset.
    """
    result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if not dataset.s3_cleaned_key:
        raise HTTPException(status_code=400, detail="Dataset has not been cleaned yet")

    # Determine S3 key based on format
    if format == "csv":
        s3_key = dataset.s3_cleaned_key.replace(".parquet", ".csv")
    else:
        s3_key = dataset.s3_cleaned_key

    # Check file exists
    meta = head_object(s3_key)
    if not meta:
        raise HTTPException(status_code=404, detail=f"Cleaned file not found in {format} format")

    # Generate presigned GET URL
    download_url, expires_at = generate_presigned_get_url(s3_key)

    return DownloadResponse(
        download_url=download_url,
        expires_at=expires_at,
        format=format,
        row_count=None,  # Could be fetched from preview table if needed
    )
