"""
NL→SQL Query API router.
"""
import uuid
import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models.dataset import Dataset
from app.models.query import QueryLog, ParquetSchemaVersion
from app.schemas.query import QueryRequest, QueryResponse
from app.services.nl_to_sql import run_nl_to_sql_pipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/datasets", tags=["queries"])


@router.post("/{dataset_id}/query", response_model=QueryResponse)
async def query_dataset(
    dataset_id: uuid.UUID,
    body: QueryRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Execute a natural language query against a cleaned dataset.
    Inline execution with 30s timeout. Full NL→SQL→validate→execute→return pipeline.
    """
    # Get dataset
    result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if not dataset.s3_cleaned_key:
        raise HTTPException(
            status_code=400,
            detail="Dataset has not been cleaned yet. Run cleaning first.",
        )

    # Get latest schema version
    result = await db.execute(
        select(ParquetSchemaVersion)
        .where(ParquetSchemaVersion.dataset_id == dataset_id)
        .order_by(ParquetSchemaVersion.version.desc())
    )
    schema_version = result.scalar_one_or_none()

    if not schema_version:
        raise HTTPException(
            status_code=400,
            detail="No schema version found. Dataset may not have been processed correctly.",
        )

    # Run NL→SQL pipeline (inline, not async)
    query_result = run_nl_to_sql_pipeline(
        question=body.question,
        dataset_id=str(dataset_id),
        s3_parquet_key=dataset.s3_cleaned_key,
        column_map=schema_version.column_map or {},
        parquet_metadata=schema_version.parquet_metadata or {},
    )

    # Log to query_logs
    query_log = QueryLog(
        dataset_id=dataset_id,
        natural_language_query=body.question,
        generated_sql=query_result.get("sql"),
        validated=query_result.get("validated", False),
        confidence_score=query_result.get("confidence"),
        execution_time_ms=query_result.get("execution_time_ms"),
        result_row_count=query_result.get("row_count"),
        error_message=query_result.get("error"),
    )
    db.add(query_log)
    await db.commit()

    # If LLM was unavailable, return 503 with Retry-After
    if query_result.get("error") == "query_unavailable":
        return JSONResponse(
            status_code=503,
            content={
                "error": "query_unavailable",
                "reason": query_result.get("reason", "LLM service unavailable"),
            },
            headers={"Retry-After": "30"},
        )

    # If SQL validation or execution failed, return the error
    if query_result.get("error") and query_result["error"] not in ("query_unavailable",):
        return QueryResponse(
            sql=query_result.get("sql"),
            results=None,
            row_count=None,
            confidence=query_result.get("confidence"),
            execution_time_ms=query_result.get("execution_time_ms"),
            assumptions=query_result.get("assumptions"),
            error=query_result.get("error"),
            reason=query_result.get("reason"),
        )

    return QueryResponse(
        sql=query_result.get("sql"),
        results=query_result.get("results"),
        row_count=query_result.get("row_count"),
        confidence=query_result.get("confidence"),
        execution_time_ms=query_result.get("execution_time_ms"),
        assumptions=query_result.get("assumptions"),
    )
