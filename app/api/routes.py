import os
import json
import logging
import tempfile
import urllib.error
import urllib.request

import httpx
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.schemas.job import (
    DatasetAnalysisResponse,
    CleanDataRequest,
    CleanDataResponse,
    FileUploadResponse,
)
from app.models.cleaned_data import CleanedData
from app.models.job import Job
from app.db.session import get_db
from app.services.csv_loader import (
    SUPPORTED_UPLOAD_EXTENSIONS,
    is_supported_upload_file,
    save_upload_file,
)
from app.services.pipeline import analyze_csv, clean_csv_with_prompt
from app.services.chat_service import (
    get_cleaned_data_row_samples,
    get_cleaned_data_row_schema,
    get_table_schema,
)
from app.services.object_storage import (
    cleaned_output_key,
    get_object_storage_service,
)
from app.services.semantic_layer import ensure_semantic_metadata
from app.core.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/upload-file", response_model=FileUploadResponse)
async def upload_file(file: UploadFile = File(...), db: AsyncSession = Depends(get_db)):
    """
    Upload a CSV/Excel file, store it in object storage, and persist the file URL.
    """
    if not is_supported_upload_file(file):
        supported_extensions = ", ".join(sorted(SUPPORTED_UPLOAD_EXTENSIONS))
        raise HTTPException(
            status_code=400,
            detail=f"Only CSV or Excel files are allowed ({supported_extensions}).",
        )

    try:
        file_id, file_url = save_upload_file(file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    db_job = Job(
        id=file_id,
        status="uploaded",
        filename=file.filename,
        file_url=file_url,
    )
    db.add(db_job)
    await db.commit()

    return FileUploadResponse(
        file_id=file_id,
        filename=file.filename,
        file_url=file_url,
        status="uploaded",
    )

@router.post("/upload-analyze-csv/{file_id}", response_model=DatasetAnalysisResponse)
async def upload_analyze_csv(file_id: str, db: AsyncSession = Depends(get_db)):
    """
    Analyze a previously uploaded CSV/Excel file by file ID.
    """
    db_job = await db.get(Job, file_id)
    if db_job is None:
        raise HTTPException(status_code=404, detail="File ID not found.")

    db_job.status = "analyzing"
    db_job.message = None
    await db.commit()

    try:
        analysis_response = await analyze_csv(file_id)
        db_job.status = "analyzed"
        db_job.analysis = analysis_response.model_dump()
        db_job.quality_score = analysis_response.quality_score
        await db.commit()
        return analysis_response
    except FileNotFoundError as exc:
        await db.rollback()
        db_job.status = "failed"
        db_job.message = str(exc)
        await db.commit()
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as e:
        await db.rollback()
        db_job.status = "failed"
        db_job.message = str(e)
        await db.commit()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/clean-csv/{job_id}", response_model=CleanDataResponse)
async def trigger_cleaning(
    job_id: str,
    request: CleanDataRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Run pure AI cleaning pipeline for a job based on a chosen prompt.
    """
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
            # Metadata enrichment is best effort; cleaning should still succeed.
            logger.exception("Failed to build semantic metadata for job_id=%s", job_id)

        return response
    except Exception as e:
        await db.rollback()
        db_job.status = "failed"
        db_job.message = str(e)
        await db.commit()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/download-cleaned/{job_id}")
async def download_cleaned(job_id: str, db: AsyncSession = Depends(get_db)):
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

@router.get("/test-ollama")
async def test_ollama_connection(prompt: str = "Reply with a short health check message."):
    """
    Verify Ollama is reachable and optionally return a small LLM reply for the given prompt.
    Echoes the provided prompt so callers can confirm which topic they sent.
    """
    tags_url = f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/tags"

    try:
        with urllib.request.urlopen(tags_url, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "ok": False,
                "message": "Could not connect to Ollama.",
                "base_url": settings.OLLAMA_BASE_URL,
                "model": settings.OLLAMA_MODEL,
                "error": str(exc),
            },
        )

    models = payload.get("models", [])
    available_models = [model.get("name", "") for model in models]
    matching_models = [
        model_name
        for model_name in available_models
        if model_name == settings.OLLAMA_MODEL or model_name.startswith(f"{settings.OLLAMA_MODEL}:")
    ]
    model_found = bool(matching_models)
    generate_model = matching_models[0] if matching_models else settings.OLLAMA_MODEL

    if not model_found:
        return {
            "ok": False,
            "base_url": settings.OLLAMA_BASE_URL,
            "model": settings.OLLAMA_MODEL,
            "model_found": False,
            "available_models": available_models,
            "prompt": prompt,
            "message": "Ollama is reachable, but the configured model was not found.",
        }

    # Try a short generation so callers can see an actual model reply.
    model_reply = None
    model_error = None
    try:
        generate_url = f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/generate"
        payload = {
            "model": generate_model,
            "prompt": prompt,
            "stream": False,
        }
        async with httpx.AsyncClient(timeout=settings.OLLAMA_TIMEOUT) as client:
            resp = await client.post(generate_url, json=payload)
            resp.raise_for_status()
            body = resp.json()
            # Ollama returns the text under "response"
            model_reply = body.get("response")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Ollama generate failed during health check: %s", exc)
        model_error = str(exc)

    return {
        "ok": True,
        "base_url": settings.OLLAMA_BASE_URL,
        "model": settings.OLLAMA_MODEL,
        "model_used": generate_model,
        "model_found": True,
        "available_models": available_models,
        "prompt": prompt,
        "model_reply": model_reply,
        "model_error": model_error,
        "message": "Ollama is reachable and generation succeeded." if model_reply is not None else "Ollama is reachable (generation skipped or failed).",
    }
