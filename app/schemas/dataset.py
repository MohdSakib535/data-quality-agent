"""
Pydantic request/response schemas for dataset endpoints.
"""
import uuid
from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field


# ── Initiate Upload ──────────────────────────────────────────
class InitiateUploadRequest(BaseModel):
    model_config = ConfigDict(strict=True)

    filename: str = Field(..., min_length=1, max_length=512)
    file_size_bytes: int = Field(..., gt=0)
    user_id: uuid.UUID


class InitiateUploadResponse(BaseModel):
    dataset_id: uuid.UUID
    job_id: uuid.UUID
    upload_url: str
    expires_at: datetime


# ── Complete Upload ──────────────────────────────────────────
class CompleteUploadResponse(BaseModel):
    dataset_id: uuid.UUID
    status: str
    analyze_job_id: uuid.UUID


# ── Analyze ──────────────────────────────────────────────────
class AnalyzeResponse(BaseModel):
    job_id: uuid.UUID
    status: str


# ── Clean ────────────────────────────────────────────────────
class CleanRequest(BaseModel):
    model_config = ConfigDict(strict=True)

    prompt: str = Field(..., min_length=10, max_length=2000)
    user_id: uuid.UUID


class CleanResponse(BaseModel):
    job_id: uuid.UUID
    status: str


# ── Download ─────────────────────────────────────────────────
class DownloadResponse(BaseModel):
    download_url: str
    expires_at: datetime
    format: str
    row_count: int | None = None


# ── Dataset Profile (returned in analysis results) ──────────
class DatasetProfileResponse(BaseModel):
    dataset_id: uuid.UUID
    row_count: int | None = None
    column_count: int | None = None
    schema_snapshot: dict | None = None
    quality_score: float | None = None
    null_stats: dict | None = None
    duplicate_count: int | None = None
    llm_suggestions: list | None = None
    profiled_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True)
