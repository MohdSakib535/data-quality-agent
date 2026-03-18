import os
import json
import urllib.error
import urllib.request
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import FileResponse
from typing import Dict, Any

from app.schemas.models import DatasetAnalysisResponse, CleanDataRequest, CleanDataResponse
from app.services.csv_loader import save_upload_file
from app.services.pipeline import analyze_csv, clean_csv_with_prompt
from app.core.config import settings

router = APIRouter()

# In-memory status tracker (for demonstration/starter code, use Redis in production)
JOB_STATUS_DB: Dict[str, Dict[str, Any]] = {}

@router.post("/upload-analyze-csv", response_model=DatasetAnalysisResponse)
async def upload_analyze_csv(file: UploadFile = File(...)):
    """
    Upload a CSV file and get an immediate AI-driven data quality analysis and suggestions.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed.")
        
    job_id = save_upload_file(file)
    JOB_STATUS_DB[job_id] = {
        "job_id": job_id,
        "status": "analyzing",
        "filename": file.filename
    }
    
    try:
        # Run synchronous analysis
        analysis_response = analyze_csv(job_id)
        JOB_STATUS_DB[job_id]["status"] = "analyzed"
        JOB_STATUS_DB[job_id]["analysis"] = analysis_response.dict()
        return analysis_response
    except Exception as e:
        JOB_STATUS_DB[job_id]["status"] = "failed"
        JOB_STATUS_DB[job_id]["message"] = str(e)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/clean-csv/{job_id}", response_model=CleanDataResponse)
def trigger_cleaning(
    job_id: str,
    request: CleanDataRequest,
    download: bool = Query(False, description="Return the cleaned CSV file directly instead of JSON metadata."),
):
    """
    Run pure AI cleaning pipeline for a job based on a chosen prompt.
    """
    if job_id not in JOB_STATUS_DB:
        file_path = os.path.join(settings.UPLOAD_DIR, f"{job_id}.csv")
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Job ID not found.")
        JOB_STATUS_DB[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "filename": os.path.basename(file_path),
        }
        
    JOB_STATUS_DB[job_id]["status"] = "processing"

    try:
        response = clean_csv_with_prompt(job_id, request.prompt)
        JOB_STATUS_DB[job_id]["status"] = "completed"
        if download:
            path = os.path.join(settings.OUTPUT_DIR, f"{job_id}_cleaned.csv")
            if not os.path.exists(path):
                raise HTTPException(status_code=404, detail="Cleaned file not found. Ensure job is completed.")
            return FileResponse(path, media_type="text/csv", filename=f"{job_id}_cleaned.csv")
        return response
    except Exception as e:
        JOB_STATUS_DB[job_id]["status"] = "failed"
        JOB_STATUS_DB[job_id]["message"] = str(e)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/download-cleaned/{job_id}")
async def download_cleaned(job_id: str):
    """Download the final cleaned CSV file."""
    path = os.path.join(settings.OUTPUT_DIR, f"{job_id}_cleaned.csv")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Cleaned file not found. Ensure job is completed.")
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
