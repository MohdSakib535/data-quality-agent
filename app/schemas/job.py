from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

# Job Status Schemas
class JobResponse(BaseModel):
    job_id: str = Field(..., alias="id", description="Unique identifier for the processing job")
    status: str = Field(..., description="Current status of the job")
    filename: Optional[str] = Field(None, description="Original uploaded filename")
    file_url: Optional[str] = Field(None, description="Stored object storage URL for the uploaded file")
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
    id: Optional[str] = Field(None, description="Unique identifier for this suggestion")
    issue_description: str = Field(description="Description of the data issue found")
    priority: str = Field(description="Priority of the issue: Low, Medium, High")
    resolution_prompt: str = Field(description="Suggested prompt to send to the AI to clean this issue")

class DatasetAnalysisPayload(BaseModel):
    quality_score: int = Field(ge=0, le=100, description="Overall quality score from 0 to 100")
    suggestions: List[DataSuggestion] = Field(default_factory=list, description="List of suggested cleaning actions")

class DatasetAnalysisResponse(BaseModel):
    job_id: str = Field(description="The job ID for this dataset")
    source_type: str = Field(default="raw", description="Whether the analysis used raw or clean data")
    quality_score: int = Field(ge=0, le=100, description="Overall quality score from 0 to 100")
    suggestions: List[DataSuggestion] = Field(default_factory=list, description="List of suggested cleaning actions")

class SuggestionDetailResponse(BaseModel):
    id: str = Field(description="Unique identifier for this suggestion")
    job_id: str = Field(description="The job/file ID this suggestion belongs to")
    source_type: str = Field(description="Whether this suggestion came from raw or clean data analysis")
    issue_description: str = Field(description="Description of the data issue found")
    priority: str = Field(description="Priority of the issue: Low, Medium, High")
    resolution_prompt: str = Field(description="Suggested prompt to send to the AI to clean this issue")
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class FileUploadResponse(BaseModel):
    file_id: str = Field(description="The stored file ID")
    filename: Optional[str] = Field(None, description="Original uploaded filename")
    file_url: str = Field(description="Stored object storage URL for the uploaded file")
    status: str = Field(description="Upload status")


class CleanedFileResponse(BaseModel):
    job_id: str = Field(description="The cleaned job ID")
    source_file_id: str = Field(description="The original uploaded file/job ID")
    status: str = Field(description="Current cleaning status")
    filename: Optional[str] = Field(None, description="Original uploaded filename")
    prompt: str = Field(description="Prompt used to generate the cleaned file")
    cleaned_file_path: str = Field(description="Stored object storage URL for the cleaned file")
    cleaned_rows: int = Field(default=0, description="Total rows written to the cleaned output")
    quality_score: Optional[int] = Field(None, description="Latest quality score for cleaned data analysis")
    analysis: Optional[Dict[str, Any]] = Field(None, description="Latest analysis payload for cleaned data")
    message: Optional[str] = Field(None, description="Additional status message or error detail")
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class CleanedFileListResponse(BaseModel):
    cleaned_files: List[CleanedFileResponse] = Field(default_factory=list, description="All cleaned files")

# User Request / Response Schemas
class CleanDataRequest(BaseModel):
    prompt: str = Field(..., description="The user's prompt instructing the AI on how to clean the data")

class CleanDataResponse(BaseModel):
    job_id: str = Field(description="The job ID")
    source_file_id: str = Field(description="The source uploaded file/job ID used for cleaning")
    status: str = Field(description="Status of the request (e.g., success, failed)")
    cleaned_file_url: str = Field(description="Endpoint URL to download the cleaned CSV")
    message: Optional[str] = Field(None, description="Additional status message for background cleaning")
    cleaned_rows: int = Field(
        default=0,
        description="Total number of rows written to the cleaned CSV output",
    )
    cleaned_data: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Preview rows from the cleaned output; large files are not returned fully in the API response",
    )


class CleanJobDetailResponse(BaseModel):
    job_id: str = Field(description="The job ID")
    source_file_id: str = Field(description="The source uploaded file/job ID used for cleaning")
    status: str = Field(description="Current cleaning status")
    prompt: Optional[str] = Field(None, description="Prompt used for the current or most recent cleaning run")
    cleaned_file_url: str = Field(description="Endpoint URL to download the cleaned CSV")
    cleaned_file_path: Optional[str] = Field(None, description="Stored object storage URL for the cleaned file")
    cleaned_rows: int = Field(default=0, description="Total number of rows written to the cleaned CSV output")
    cleaned_data: List[Dict[str, Any]] = Field(default_factory=list, description="Preview rows from the cleaned output")
    quality_score: Optional[int] = Field(None, description="Latest quality score for cleaned data analysis")
    analysis: Optional[Dict[str, Any]] = Field(None, description="Latest analysis payload for cleaned data")
    message: Optional[str] = Field(None, description="Additional status message or error detail")
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
