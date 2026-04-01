"""
Pydantic schemas for NL→SQL query endpoint.
"""
import uuid
from pydantic import BaseModel, ConfigDict, Field


class QueryRequest(BaseModel):
    model_config = ConfigDict(strict=True)

    question: str = Field(..., min_length=3, max_length=2000)
    user_id: uuid.UUID


class QueryResponse(BaseModel):
    sql: str | None = None
    results: list[dict] | None = None
    row_count: int | None = None
    confidence: float | None = None
    execution_time_ms: int | None = None
    assumptions: list[str] | None = None
    error: str | None = None
    reason: str | None = None
