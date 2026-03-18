from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

# Job Status Schemas
class JobResponse(BaseModel):
    job_id: str = Field(..., alias="id", description="Unique identifier for the processing job")
    status: str = Field(..., description="Current status of the job")
    filename: Optional[str] = Field(None, description="Original uploaded filename")
    message: Optional[str] = Field(None, description="Additional status message or error detail")
    analysis: Optional[Dict[str, Any]] = Field(None, description="Generated output metadata for completed jobs")
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    class Config:
        populate_by_name = True
        from_attributes = True

class JobListResponse(BaseModel):
    jobs: List[JobResponse] = Field(default_factory=list, description="All tracked jobs")

# AI Analysis Schemas
class DataSuggestion(BaseModel):
    issue_description: str = Field(description="Description of the data issue found")
    priority: str = Field(description="Priority of the issue: Low, Medium, High")
    resolution_prompt: str = Field(description="Suggested prompt to send to the AI to clean this issue")

class DatasetAnalysisPayload(BaseModel):
    quality_score: int = Field(ge=0, le=100, description="Overall quality score from 0 to 100")
    suggestions: List[DataSuggestion] = Field(default_factory=list, description="List of suggested cleaning actions")

class DatasetAnalysisResponse(BaseModel):
    job_id: str = Field(description="The job ID for this dataset")
    quality_score: int = Field(ge=0, le=100, description="Overall quality score from 0 to 100")
    suggestions: List[DataSuggestion] = Field(default_factory=list, description="List of suggested cleaning actions")

# User Request / Response Schemas
class CleanDataRequest(BaseModel):
    prompt: str = Field(..., description="The user's prompt instructing the AI on how to clean the data")

class CleanDataResponse(BaseModel):
    job_id: str = Field(description="The job ID")
    status: str = Field(description="Status of the request (e.g., success, failed)")
    cleaned_file_url: str = Field(description="Endpoint URL to download the cleaned CSV")
    cleaned_rows: int = Field(
        default=0,
        description="Total number of rows written to the cleaned CSV output",
    )
    cleaned_data: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Preview rows from the cleaned output; large files are not returned fully in the API response",
    )
