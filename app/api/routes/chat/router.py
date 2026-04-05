import asyncio
import logging
import re
import time
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.session import engine, get_db
from app.models.chat_history import ChatHistory
from app.services.chat_service import (
    explain_query,
    generate_sql_via_ollama,
    get_cleaned_data_row_samples,
    get_cleaned_data_row_schema,
    get_table_schema,
    plan_query_via_ollama,
    repair_sql_via_ollama,
    resolve_ollama_host,
    safe_execute,
    save_chat_history,
)
from app.services.semantic_layer import (
    ensure_semantic_metadata,
    retrieve_relevant_semantic_context,
    semantic_metadata_exists,
)
from app.utils.chat_utils import (
    JobNotFoundError,
    OllamaResponseError,
    OllamaUnavailableError,
    format_execution_time,
    sanitize_question,
    validate_sql,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])
CHAT_RESPONSE_EXCLUDED_FIELDS = {"prompt", "cleaned_file_path"}


class ChatQueryRequest(BaseModel):
    job_id: str
    question: str


class ChatQueryResponse(BaseModel):
    job_id: str
    question: str
    sql_generated: str
    data: list[dict[str, Any]]
    row_count: int
    execution_time_ms: float
    error: str | None


class ChatSchemaResponse(BaseModel):
    job_id: str
    table_name: str
    columns: list[dict[str, str]]


class ChatHistoryResponse(BaseModel):
    job_id: str
    history: list[dict[str, Any]]
    total: int


class ChatHealthResponse(BaseModel):
    status: str
    db: bool
    ollama: bool


def _normalize_job_id(job_id: str) -> str:
    cleaned_job_id = sanitize_question(job_id)
    return re.sub(r"[^a-zA-Z0-9_-]", "", cleaned_job_id)


async def _resolve_job_table_name(job_id: str) -> str:
    normalized_job_id = _normalize_job_id(job_id)
    if not normalized_job_id:
        raise JobNotFoundError("Job ID not found.")

    async with engine.connect() as conn:
        cleaned_data_result = await conn.execute(
            text("SELECT 1 FROM cleaned_data WHERE job_id = :job_id LIMIT 1"),
            {"job_id": normalized_job_id},
        )
        if cleaned_data_result.fetchone() is None:
            raise JobNotFoundError("No cleaned data found for this job_id.")

    return "cleaned_data"


def _ensure_job_filter(sql: str, job_id: str) -> str:
    normalized_sql = sql.strip().rstrip(";")
    upper_sql = normalized_sql.upper()

    if "JOB_ID" in upper_sql:
        return normalized_sql

    clause = f"job_id = '{job_id}'"
    split_match = re.search(
        r"\b(ORDER\s+BY|GROUP\s+BY|LIMIT|OFFSET|FETCH)\b",
        normalized_sql,
        flags=re.IGNORECASE,
    )

    if split_match:
        before = normalized_sql[:split_match.start()].rstrip()
        after = normalized_sql[split_match.start():].lstrip()
    else:
        before = normalized_sql
        after = ""

    if re.search(r"\bWHERE\b", before, flags=re.IGNORECASE):
        scoped = f"{before} AND {clause}"
    else:
        scoped = f"{before} WHERE {clause}"

    if after:
        return f"{scoped} {after}"
    return scoped


def _needs_json_row_expansion(table_name: str, sql: str) -> bool:
    if table_name != "cleaned_data":
        return False

    upper_sql = sql.upper()
    return "JSONB_ARRAY_ELEMENTS" not in upper_sql


@router.post("/query", response_model=ChatQueryResponse)
async def chat_query(request: ChatQueryRequest):
    start = time.time()
    job_id = _normalize_job_id(request.job_id)
    question = sanitize_question(request.question)

    if not job_id:
        raise HTTPException(status_code=400, detail="job_id is required.")
    if not question:
        raise HTTPException(status_code=400, detail="question is required.")

    try:
        table_name = await _resolve_job_table_name(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    schema = await get_table_schema(table_name)
    if not schema:
        raise HTTPException(status_code=400, detail="Table schema is empty.")
    row_schema: list[dict[str, str]] = []
    row_samples: list[dict[str, Any]] = []
    if table_name == "cleaned_data":
        row_schema = await get_cleaned_data_row_schema(job_id)
        row_samples = await get_cleaned_data_row_samples(
            job_id,
            sample_items=settings.SEMANTIC_ROW_SAMPLE_LIMIT,
        )

    if not await semantic_metadata_exists(job_id=job_id, table_name=table_name):
        await ensure_semantic_metadata(
            job_id=job_id,
            table_name=table_name,
            table_schema=schema,
            row_schema=row_schema,
            row_samples=row_samples,
        )
    semantic_context = await retrieve_relevant_semantic_context(
        job_id=job_id,
        table_name=table_name,
        question=question,
        limit=settings.SEMANTIC_TOP_COLUMNS,
    )

    sql_generated = ""
    query_plan: dict[str, Any] = {}
    try:
        scoped_question = (
            f"{question}\n\n"
            f"Only return rows for job_id = '{job_id}'."
        )
        try:
            query_plan = await plan_query_via_ollama(
                scoped_question,
                schema,
                table_name,
                row_schema=row_schema,
                semantic_context=semantic_context,
            )
        except OllamaResponseError:
            logger.warning(
                "Planner returned non-JSON payload for job_id=%s; continuing without plan.",
                job_id,
            )
            query_plan = {}

        confidence_raw = query_plan.get("confidence", 0.0)
        try:
            planner_confidence = float(confidence_raw or 0.0)
        except (TypeError, ValueError):
            planner_confidence = 0.0

        if (
            query_plan.get("needs_clarification")
            and query_plan.get("clarification_question")
            and planner_confidence < 0.45
        ):
            clarification = str(query_plan["clarification_question"])
            asyncio.create_task(
                save_chat_history(
                    job_id=job_id,
                    question=question,
                    sql_generated="",
                    row_count=0,
                    success=False,
                    error_message=f"Need clarification: {clarification}",
                )
            )
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "Need clarification to answer this question accurately.",
                    "clarification_question": clarification,
                },
            )

        sql_generated = await generate_sql_via_ollama(
            scoped_question,
            schema,
            table_name,
            row_schema=row_schema,
            semantic_context=semantic_context,
            query_plan=query_plan,
        )
    except OllamaUnavailableError:
        asyncio.create_task(
            save_chat_history(
                job_id=job_id,
                question=question,
                sql_generated=sql_generated,
                row_count=0,
                success=False,
                error_message="Ollama service unavailable.",
            )
        )
        raise HTTPException(status_code=503, detail="Ollama service is unavailable.")
    except OllamaResponseError:
        asyncio.create_task(
            save_chat_history(
                job_id=job_id,
                question=question,
                sql_generated=sql_generated,
                row_count=0,
                success=False,
                error_message="Invalid response from Ollama.",
            )
        )
        raise HTTPException(status_code=422, detail="Failed to generate valid SQL.")

    is_valid_sql, reason = validate_sql(sql_generated)
    if not is_valid_sql:
        asyncio.create_task(
            save_chat_history(
                job_id=job_id,
                question=question,
                sql_generated=sql_generated,
                row_count=0,
                success=False,
                error_message=reason,
            )
        )
        raise HTTPException(status_code=400, detail=f"Invalid SQL: {reason}")

    sql_generated = _ensure_job_filter(sql_generated, job_id)

    max_rows = settings.CHAT_MAX_ROWS

    if _needs_json_row_expansion(table_name, sql_generated):
        try:
            repaired_sql = await repair_sql_via_ollama(
                question=question,
                schema=schema,
                table_name=table_name,
                failed_sql=sql_generated,
                db_error=(
                    "Query must analyze rows inside cleaned_data JSON array using "
                    "jsonb_array_elements(cleaned_data::jsonb). "
                    "Top-level cleaned_data row query is not sufficient."
                ),
                row_schema=row_schema,
                semantic_context=semantic_context,
                query_plan=query_plan,
            )
            is_repaired_valid, repaired_reason = validate_sql(repaired_sql)
            if not is_repaired_valid:
                raise ValueError(f"Invalid repaired SQL: {repaired_reason}")
            sql_generated = _ensure_job_filter(repaired_sql, job_id)
        except Exception:
            asyncio.create_task(
                save_chat_history(
                    job_id=job_id,
                    question=question,
                    sql_generated=sql_generated,
                    row_count=0,
                    success=False,
                    error_message="Failed to produce analytical SQL for cleaned_data.",
                )
            )
            raise HTTPException(
                status_code=422,
                detail="Could not generate analytical SQL for this question. Try rephrasing with exact column intent.",
            )

    try:
        async with engine.connect() as conn:
            await explain_query(conn, sql_generated)
            data = await safe_execute(conn, sql_generated, max_rows=max_rows)
    except Exception as exec_error:
        logger.exception("Failed SQL execution for job_id=%s", job_id)

        try:
            repaired_sql = await repair_sql_via_ollama(
                question=question,
                schema=schema,
                table_name=table_name,
                failed_sql=sql_generated,
                db_error=str(exec_error),
                row_schema=row_schema,
                semantic_context=semantic_context,
                query_plan=query_plan,
            )
            is_repaired_valid, repaired_reason = validate_sql(repaired_sql)
            if not is_repaired_valid:
                raise ValueError(f"Invalid repaired SQL: {repaired_reason}")

            repaired_sql = _ensure_job_filter(repaired_sql, job_id)
            async with engine.connect() as conn:
                await explain_query(conn, repaired_sql)
                data = await safe_execute(conn, repaired_sql, max_rows=max_rows)
            sql_generated = repaired_sql
        except Exception:
            asyncio.create_task(
                save_chat_history(
                    job_id=job_id,
                    question=question,
                    sql_generated=sql_generated,
                    row_count=0,
                    success=False,
                    error_message="Database execution failed after SQL repair attempt.",
                )
            )
            raise HTTPException(
                status_code=500,
                detail="Failed to execute SQL query. Try simplifying or rephrasing the question.",
            )

    filtered_data = [
        {
            key: value
            for key, value in row.items()
            if key not in CHAT_RESPONSE_EXCLUDED_FIELDS
        }
        for row in data
    ]

    row_count = len(filtered_data)
    asyncio.create_task(
        save_chat_history(
            job_id=job_id,
            question=question,
            sql_generated=sql_generated,
            row_count=row_count,
            success=True,
            error_message=None,
        )
    )

    return ChatQueryResponse(
        job_id=job_id,
        question=question,
        sql_generated=sql_generated,
        data=filtered_data,
        row_count=row_count,
        execution_time_ms=format_execution_time(start),
        error=None,
    )


@router.get("/schema/{job_id}", response_model=ChatSchemaResponse)
async def get_job_schema(job_id: str):
    normalized_job_id = _normalize_job_id(job_id)
    if not normalized_job_id:
        raise HTTPException(status_code=400, detail="job_id is required.")

    try:
        table_name = await _resolve_job_table_name(normalized_job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    schema = await get_table_schema(table_name)
    if not schema:
        raise HTTPException(status_code=400, detail="Table schema is empty.")

    return ChatSchemaResponse(
        job_id=normalized_job_id,
        table_name=table_name,
        columns=schema,
    )


@router.get("/history/{job_id}", response_model=ChatHistoryResponse)
async def get_chat_history(
    job_id: str,
    limit: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    normalized_job_id = _normalize_job_id(job_id)
    if not normalized_job_id:
        raise HTTPException(status_code=400, detail="job_id is required.")

    history_result = await db.execute(
        select(ChatHistory)
        .where(ChatHistory.job_id == normalized_job_id)
        .order_by(ChatHistory.created_at.desc())
        .limit(limit)
    )
    history_rows = history_result.scalars().fetchmany(limit)

    total_result = await db.execute(
        select(func.count())
        .select_from(ChatHistory)
        .where(ChatHistory.job_id == normalized_job_id)
    )
    total = int(total_result.scalar() or 0)

    history_payload = [
        {
            "id": row.id,
            "question": row.question,
            "sql_generated": row.sql_generated,
            "row_count": row.row_count,
            "success": row.success,
            "error_message": row.error_message,
            "created_at": row.created_at.isoformat() if row.created_at else None,
        }
        for row in history_rows
    ]

    return ChatHistoryResponse(
        job_id=normalized_job_id,
        history=history_payload,
        total=total,
    )


@router.get("/health", response_model=ChatHealthResponse)
async def chat_health():
    db_ok = False
    ollama_ok = False

    try:
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT 1"))
            db_ok = result.fetchone() is not None
    except Exception:
        logger.exception("Chat health DB check failed.")

    ollama_host = resolve_ollama_host()
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{ollama_host}/api/tags")
            ollama_ok = response.status_code == 200
    except httpx.HTTPError:
        ollama_ok = False

    return ChatHealthResponse(
        status="ok" if db_ok and ollama_ok else "degraded",
        db=db_ok,
        ollama=ollama_ok,
    )
