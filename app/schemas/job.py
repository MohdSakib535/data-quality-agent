"""
Pydantic schemas for job status endpoint.
"""
import uuid
from datetime import datetime
from pydantic import BaseModel, ConfigDict


class JobStatusResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    dataset_id: uuid.UUID
    job_type: str
    status: str
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_message: str | None = None
    retry_count: int = 0
    metadata: dict = {}
    warning: str | None = None  # populated when status == "degraded"
