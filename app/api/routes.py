import os
import json
import urllib.error
import urllib.request
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.schemas.job import DatasetAnalysisResponse, CleanDataRequest, CleanDataResponse
from app.models.cleaned_data import CleanedData
from app.models.job import Job
from app.db.session import get_db
from app.services.csv_loader import save_upload_file
from app.services.pipeline import analyze_csv, clean_csv_with_prompt
from app.core.config import settings

router = APIRouter()

@router.post("/upload-analyze-csv", response_model=DatasetAnalysisResponse)
async def upload_analyze_csv(file: UploadFile = File(...), db: AsyncSession = Depends(get_db)):
    """
    Upload a CSV file and get an immediate AI-driven data quality analysis and suggestions.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed.")
        
    job_id = save_upload_file(file)
    
    # Create DB entry
    db_job = Job(id=job_id, status="analyzing", filename=file.filename)
    db.add(db_job)
    await db.commit()
    
    try:
        # Run asynchronous analysis
        analysis_response = await analyze_csv(job_id)
        db_job.status = "analyzed"
        db_job.analysis = analysis_response.model_dump()
        db_job.quality_score = analysis_response.quality_score
        await db.commit()
        return analysis_response
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
        file_path = os.path.join(settings.UPLOAD_DIR, f"{job_id}.csv")
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Job ID not found.")
        db_job = Job(id=job_id, status="queued", filename=os.path.basename(file_path))
        db.add(db_job)
        await db.commit()
        
    db_job.status = "processing"
    await db.commit()

    try:
        response = await clean_csv_with_prompt(job_id, request.prompt)
        cleaned_result = await db.get(CleanedData, job_id)

        if cleaned_result is None:
            cleaned_result = CleanedData(
                job_id=job_id,
                prompt=request.prompt,
                cleaned_file_path=os.path.join(settings.OUTPUT_DIR, f"{job_id}_cleaned.csv"),
                cleaned_data=response.cleaned_data,
            )
            db.add(cleaned_result)
        else:
            cleaned_result.prompt = request.prompt
            cleaned_result.cleaned_file_path = os.path.join(settings.OUTPUT_DIR, f"{job_id}_cleaned.csv")
            cleaned_result.cleaned_data = response.cleaned_data

        db_job.status = "completed"
        db_job.message = None
        await db.commit()
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

    path = cleaned_result.cleaned_file_path
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Cleaned file not found. Run cleaning again.")
    return FileResponse(path, media_type='text/csv', filename=f"{job_id}_cleaned.csv")

@router.get("/test-ollama")
async def test_ollama_connection(prompt: str = "Reply with a short health check message."):
    """Verify Ollama is reachable."""
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
    model_found = any(
        model_name == settings.OLLAMA_MODEL or model_name.startswith(f"{settings.OLLAMA_MODEL}:")
        for model_name in available_models
    )

    if not model_found:
        return {
            "ok": False,
            "base_url": settings.OLLAMA_BASE_URL,
            "model": settings.OLLAMA_MODEL,
            "model_found": False,
            "available_models": available_models,
            "message": "Ollama is reachable, but the configured model was not found.",
        }

    return {
        "ok": True,
        "base_url": settings.OLLAMA_BASE_URL,
        "model": settings.OLLAMA_MODEL,
        "model_found": True,
        "available_models": available_models,
        "message": "Ollama is reachable and generation succeeded.",
    }
