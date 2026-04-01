"""
NL → SQL pipeline.
Load schema → call LLM → validate SQL → execute via DuckDB → map columns back.
"""
import json
import logging
import time
from typing import Any

import duckdb

from app.core.config import settings
from app.services.llm import call_llm_with_fallback
from app.services.sql_validator import validate_sql

logger = logging.getLogger(__name__)


def _build_column_info(column_map: dict, parquet_metadata: dict) -> list[dict]:
    """Build column info for LLM prompt from schema version data."""
    columns = []
    col_types = parquet_metadata.get("column_types", {})
    sample_values = parquet_metadata.get("sample_values", {})

    for cleaned_name, original_name in column_map.items():
        columns.append({
            "name": cleaned_name,
            "original_name": original_name,
            "type": col_types.get(cleaned_name, "unknown"),
            "sample_values": sample_values.get(cleaned_name, []),
        })
    return columns


def generate_sql_from_question(
    question: str,
    column_map: dict,
    parquet_metadata: dict,
    dataset_id: str,
) -> dict[str, Any]:
    """
    Call LLM to generate SQL from a natural language question.
    Uses call_llm_with_fallback — never calls LLM directly.
    """
    columns = _build_column_info(column_map, parquet_metadata)

    prompt = f"""Convert this natural language question to a SQL SELECT query.

TABLE: dataset
COLUMNS:
{json.dumps(columns, indent=2)}

QUESTION: {question}

RULES:
- Return ONLY valid SQL SELECT
- Use table name 'dataset'
- Add LIMIT 1000 unless user specifies otherwise
- Use the cleaned column names (not original names) in the SQL
- Only reference columns that exist in the schema

Return ONLY this JSON:
{{
    "sql": "SELECT ... FROM dataset WHERE ... LIMIT 1000",
    "assumptions": ["list of assumptions made"],
    "confidence": 0.85
}}"""

    fallback = {
        "error": "query_unavailable",
        "reason": "LLM service unavailable",
    }

    result = call_llm_with_fallback(
        prompt=prompt,
        expected_keys=["sql"],
        context="nl_to_sql",
        fallback_response=fallback,
    )

    return {
        "llm_result": result.data,
        "llm_assisted": result.llm_assisted,
        "tier_used": result.tier_used,
        "warnings": result.warnings,
    }


def execute_query_on_parquet(
    s3_parquet_key: str,
    safe_sql: str,
    timeout: int | None = None,
) -> tuple[list[dict], int]:
    """
    Execute validated SQL against a Parquet file via DuckDB.
    Uses DuckDB's httpfs extension for S3 access.
    Returns (results_as_dicts, execution_time_ms).
    """
    timeout = timeout or settings.QUERY_TIMEOUT
    start = time.time()

    # Build the S3 path for DuckDB
    # Convert internal endpoint to DuckDB-compatible path
    s3_path = f"s3://{settings.S3_BUCKET_NAME}/{s3_parquet_key}"

    conn = duckdb.connect(":memory:", read_only=False)
    try:
        # Configure S3 access for DuckDB
        conn.execute("INSTALL httpfs;")
        conn.execute("LOAD httpfs;")
        conn.execute(f"SET s3_region='{settings.S3_REGION}';")
        conn.execute(f"SET s3_access_key_id='{settings.S3_ACCESS_KEY_ID}';")
        conn.execute(f"SET s3_secret_access_key='{settings.S3_SECRET_ACCESS_KEY}';")
        conn.execute(f"SET s3_endpoint='{settings.S3_ENDPOINT_URL.replace('http://', '').replace('https://', '')}';")
        conn.execute("SET s3_use_ssl=false;")
        conn.execute("SET s3_url_style='path';")

        # Create view from Parquet
        conn.execute(f"CREATE VIEW dataset AS SELECT * FROM read_parquet('{s3_path}');")

        # Execute the validated SQL with timeout
        conn.execute(f"SET statement_timeout='{timeout}s';")
        result = conn.execute(safe_sql)

        # Fetch results
        columns = [desc[0] for desc in result.description]
        rows = result.fetchmany(settings.QUERY_MAX_ROWS)

        results = [dict(zip(columns, row)) for row in rows]
        execution_time_ms = int((time.time() - start) * 1000)

        logger.info(
            "DuckDB query executed",
            extra={
                "s3_key": s3_parquet_key,
                "row_count": len(results),
                "execution_time_ms": execution_time_ms,
            },
        )
        return results, execution_time_ms

    except duckdb.Error as exc:
        execution_time_ms = int((time.time() - start) * 1000)
        logger.error(
            "DuckDB query failed",
            extra={"error": str(exc), "sql": safe_sql[:200], "execution_time_ms": execution_time_ms},
        )
        raise
    finally:
        conn.close()


def map_columns_back(results: list[dict], column_map: dict) -> list[dict]:
    """
    Map cleaned column names back to original human-readable names.
    column_map: {cleaned_name: original_name}
    """
    if not column_map or not results:
        return results

    mapped = []
    for row in results:
        mapped_row = {}
        for key, value in row.items():
            original_name = column_map.get(key, key)
            mapped_row[original_name] = value
        mapped.append(mapped_row)
    return mapped


def run_nl_to_sql_pipeline(
    question: str,
    dataset_id: str,
    s3_parquet_key: str,
    column_map: dict,
    parquet_metadata: dict,
) -> dict[str, Any]:
    """
    Full NL → SQL pipeline:
    1. Generate SQL from question via LLM (with fallback)
    2. Validate SQL via sqlglot AST
    3. Execute via DuckDB on Parquet
    4. Map column names back to originals
    """
    start = time.time()

    # Step 1: Generate SQL
    gen_result = generate_sql_from_question(
        question=question,
        column_map=column_map,
        parquet_metadata=parquet_metadata,
        dataset_id=dataset_id,
    )

    llm_data = gen_result["llm_result"]

    # Check if LLM fallback returned an error
    if not gen_result["llm_assisted"] or "error" in llm_data:
        return {
            "sql": None,
            "results": None,
            "row_count": None,
            "confidence": None,
            "execution_time_ms": int((time.time() - start) * 1000),
            "assumptions": None,
            "error": llm_data.get("error", "query_unavailable"),
            "reason": llm_data.get("reason", "LLM service unavailable"),
        }

    generated_sql = llm_data.get("sql", "")
    confidence = llm_data.get("confidence", 0.0)
    assumptions = llm_data.get("assumptions", [])

    # Step 2: Validate SQL
    validation = validate_sql(generated_sql, dataset_id=dataset_id)

    if not validation.is_valid:
        logger.warning(
            "SQL validation failed",
            extra={"dataset_id": dataset_id, "error": validation.error},
        )
        return {
            "sql": generated_sql,
            "results": None,
            "row_count": None,
            "confidence": confidence,
            "execution_time_ms": int((time.time() - start) * 1000),
            "assumptions": assumptions,
            "error": "sql_validation_failed",
            "reason": validation.error,
            "validated": False,
        }

    # Step 3: Execute via DuckDB
    try:
        raw_results, exec_time = execute_query_on_parquet(
            s3_parquet_key=s3_parquet_key,
            safe_sql=validation.safe_sql,
        )
    except Exception as exc:
        return {
            "sql": validation.safe_sql,
            "results": None,
            "row_count": None,
            "confidence": confidence,
            "execution_time_ms": int((time.time() - start) * 1000),
            "assumptions": assumptions,
            "error": "query_execution_failed",
            "reason": str(exc),
            "validated": True,
        }

    # Step 4: Map columns back
    mapped_results = map_columns_back(raw_results, column_map)

    total_time = int((time.time() - start) * 1000)
    return {
        "sql": validation.safe_sql,
        "results": mapped_results,
        "row_count": len(mapped_results),
        "confidence": confidence,
        "execution_time_ms": total_time,
        "assumptions": assumptions,
        "error": None,
        "reason": None,
        "validated": True,
    }
