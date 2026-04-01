"""
Cleaning pipeline service.
5-step pipeline: deterministic clean → LLM plan → execute plan → save parquet → store preview.
Polars-first with pandas fallback. Chunk-based processing. Redis cache by dataset_id + prompt hash.
"""
import io
import json
import hashlib
import logging
import re
from datetime import datetime, timezone
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import redis

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

import pandas as pd

from app.services.llm import call_llm_with_fallback
from app.services.storage import (
    download_object_to_bytes, upload_bytes, upload_fileobj,
)
from app.utils.hashing import schema_fingerprint, prompt_cache_key
from app.utils.parquet_meta import write_parquet_with_metadata
from app.core.config import settings

logger = logging.getLogger(__name__)

# Redis client
_redis: redis.Redis | None = None


def _get_redis() -> redis.Redis:
    global _redis
    if _redis is None:
        _redis = redis.from_url(settings.REDIS_URL, decode_responses=True)
    return _redis


# ── Null tokens to normalize ─────────────────────────────────
NULL_TOKENS = {"", "n/a", "na", "null", "none", "nan", "nil", "missing",
               "not available", "not provided", "blank", "#n/a", "tbd",
               "n.a.", "-", "--", "NULL", "None"}


# ═══════════════════════════════════════════════════════════════
# STEP 1: Deterministic Cleaning
# ═══════════════════════════════════════════════════════════════

def _standardize_column_names(columns: list[str]) -> dict[str, str]:
    """Return mapping of original → cleaned column names."""
    mapping = {}
    for col in columns:
        cleaned = re.sub(r"[^\w]", "_", col.strip().lower())
        cleaned = re.sub(r"_+", "_", cleaned).strip("_")
        if not cleaned:
            cleaned = f"column_{columns.index(col)}"
        mapping[col] = cleaned
    return mapping


def deterministic_clean_polars(df: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    """Run deterministic cleaning using Polars. Returns (cleaned_df, steps_applied)."""
    steps = []

    # 1. Standardize column names
    col_map = _standardize_column_names(df.columns)
    df = df.rename(col_map)
    steps.append("Standardized column names to lowercase with underscores")

    # 2. Trim whitespace from string columns
    str_cols = [col for col, dtype in zip(df.columns, df.dtypes) if dtype == pl.Utf8]
    for col in str_cols:
        df = df.with_columns(pl.col(col).str.strip_chars().alias(col))
    if str_cols:
        steps.append(f"Trimmed whitespace from {len(str_cols)} string columns")

    # 3. Normalize null tokens
    for col in str_cols:
        df = df.with_columns(
            pl.when(pl.col(col).str.to_lowercase().is_in(list(NULL_TOKENS)))
            .then(None)
            .otherwise(pl.col(col))
            .alias(col)
        )
    steps.append("Normalized null tokens (N/A, null, None, etc.) → NULL")

    # 4. Remove fully duplicate rows
    original_len = len(df)
    df = df.unique(maintain_order=True)
    removed = original_len - len(df)
    if removed > 0:
        steps.append(f"Removed {removed} fully duplicate rows")

    # 5. Infer and cast numeric columns
    for col in str_cols:
        if col in df.columns:
            try:
                # Try casting to numeric
                numeric_col = df[col].cast(pl.Float64, strict=False)
                non_null_original = df[col].drop_nulls().len()
                non_null_cast = numeric_col.drop_nulls().len()
                if non_null_original > 0 and non_null_cast / non_null_original > 0.9:
                    df = df.with_columns(numeric_col.alias(col))
                    # Check if all values are integers
                    if all(v == int(v) for v in numeric_col.drop_nulls().to_list() if v is not None):
                        df = df.with_columns(pl.col(col).cast(pl.Int64, strict=False).alias(col))
            except Exception:
                pass

    steps.append("Inferred and cast numeric columns where safe")

    return df, steps


def deterministic_clean_pandas(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Fallback: deterministic cleaning using pandas."""
    steps = []

    # 1. Standardize column names
    col_map = _standardize_column_names(list(df.columns))
    df = df.rename(columns=col_map)
    steps.append("Standardized column names to lowercase with underscores")

    # 2. Trim whitespace
    str_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip()
    if str_cols:
        steps.append(f"Trimmed whitespace from {len(str_cols)} string columns")

    # 3. Normalize null tokens
    for col in str_cols:
        df[col] = df[col].apply(lambda x: None if str(x).strip().lower() in NULL_TOKENS else x)
    steps.append("Normalized null tokens → NULL")

    # 4. Remove duplicates
    original_len = len(df)
    df = df.drop_duplicates(keep="first").reset_index(drop=True)
    removed = original_len - len(df)
    if removed > 0:
        steps.append(f"Removed {removed} fully duplicate rows")

    # 5. Infer numeric
    for col in str_cols:
        try:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        except Exception:
            pass
    steps.append("Inferred numeric columns where safe")

    return df, steps


# ═══════════════════════════════════════════════════════════════
# STEP 2: LLM Cleaning Plan Generation
# ═══════════════════════════════════════════════════════════════

def generate_cleaning_plan(
    schema: dict[str, str],
    sample_rows: list[dict],
    user_prompt: str,
    detected_issues: list | None = None,
) -> dict[str, Any]:
    """
    Ask LLM to generate a cleaning plan as structured JSON.
    Uses call_llm_with_fallback — never calls LLM directly.
    """
    prompt = f"""You are a data cleaning assistant. Given the dataset schema, sample rows, and user instructions,
generate a cleaning plan as STRICT JSON.

SCHEMA (column_name: inferred_type):
{json.dumps(schema, indent=2)}

SAMPLE ROWS (first 50 rows after deterministic cleaning):
{json.dumps(sample_rows[:50], indent=2, default=str)}

USER INSTRUCTION:
{user_prompt}

DETECTED ISSUES:
{json.dumps(detected_issues or [], indent=2, default=str)}

Return ONLY this JSON structure — no prose, no markdown:
{{
    "steps": [
        {{"step_id": "1", "description": "...", "target_columns": ["..."]}}
    ],
    "transformations": [
        {{
            "type": "rename|fill_nulls|filter_rows|map_values|split_column|merge_columns|custom",
            "column": "column_name",
            "params": {{}}
        }}
    ],
    "assumptions": ["..."],
    "skip_llm": false
}}"""

    fallback = {
        "steps": [{"step_id": "fallback", "description": "Deterministic cleaning only"}],
        "transformations": [],
        "assumptions": ["LLM unavailable — only standard cleaning applied"],
        "skip_llm": True,
    }

    result = call_llm_with_fallback(
        prompt=prompt,
        expected_keys=["steps", "transformations"],
        context="cleaning",
        fallback_response=fallback,
        schema_fingerprint=schema_fingerprint(schema),
    )

    return {
        "plan": result.data,
        "llm_assisted": result.llm_assisted,
        "tier_used": result.tier_used,
        "warnings": result.warnings,
    }


# ═══════════════════════════════════════════════════════════════
# STEP 3: Execute Cleaning Plan
# ═══════════════════════════════════════════════════════════════

def _execute_transformation(df: pl.DataFrame, transformation: dict) -> pl.DataFrame:
    """Execute a single transformation from the LLM plan using Polars."""
    t_type = transformation.get("type", "")
    column = transformation.get("column", "")
    params = transformation.get("params", {})

    try:
        if t_type == "rename" and column and "new_name" in params:
            new_name = re.sub(r"[^\w]", "_", params["new_name"].strip().lower())
            if column in df.columns:
                df = df.rename({column: new_name})

        elif t_type == "fill_nulls" and column:
            fill_value = params.get("value")
            strategy = params.get("strategy", "literal")
            if column in df.columns:
                if strategy == "mean":
                    df = df.with_columns(pl.col(column).fill_null(pl.col(column).mean()))
                elif strategy == "median":
                    df = df.with_columns(pl.col(column).fill_null(pl.col(column).median()))
                elif strategy == "forward":
                    df = df.with_columns(pl.col(column).forward_fill())
                elif fill_value is not None:
                    df = df.with_columns(pl.col(column).fill_null(pl.lit(fill_value)))

        elif t_type == "filter_rows":
            condition = params.get("condition", "")
            col_name = params.get("column", column)
            operator = params.get("operator", "")
            value = params.get("value")

            if col_name in df.columns and operator and value is not None:
                if operator == "==":
                    df = df.filter(pl.col(col_name) == value)
                elif operator == "!=":
                    df = df.filter(pl.col(col_name) != value)
                elif operator == ">":
                    df = df.filter(pl.col(col_name) > value)
                elif operator == "<":
                    df = df.filter(pl.col(col_name) < value)
                elif operator == ">=":
                    df = df.filter(pl.col(col_name) >= value)
                elif operator == "<=":
                    df = df.filter(pl.col(col_name) <= value)
                elif operator == "not_null":
                    df = df.filter(pl.col(col_name).is_not_null())

        elif t_type == "map_values" and column:
            mapping = params.get("mapping", {})
            if column in df.columns and mapping:
                df = df.with_columns(
                    pl.col(column).cast(pl.Utf8).replace(mapping).alias(column)
                )

        elif t_type == "split_column" and column:
            delimiter = params.get("delimiter", ",")
            new_columns = params.get("new_columns", [])
            if column in df.columns and new_columns:
                for i, new_col in enumerate(new_columns):
                    safe_name = re.sub(r"[^\w]", "_", new_col.strip().lower())
                    df = df.with_columns(
                        pl.col(column).str.split(delimiter).list.get(i).alias(safe_name)
                    )

        elif t_type == "merge_columns":
            columns_to_merge = params.get("columns", [])
            separator = params.get("separator", " ")
            new_name = params.get("new_name", "merged_column")
            safe_name = re.sub(r"[^\w]", "_", new_name.strip().lower())
            existing_cols = [c for c in columns_to_merge if c in df.columns]
            if existing_cols:
                df = df.with_columns(
                    pl.concat_str([pl.col(c).cast(pl.Utf8) for c in existing_cols], separator=separator).alias(safe_name)
                )

    except Exception as exc:
        logger.warning(
            "Transformation failed, skipping",
            extra={"type": t_type, "column": column, "error": str(exc)},
        )

    return df


def execute_cleaning_plan(df: pl.DataFrame, plan: dict) -> pl.DataFrame:
    """Execute all transformations in the LLM cleaning plan."""
    transformations = plan.get("transformations", [])
    for transformation in transformations:
        df = _execute_transformation(df, transformation)
    return df


# ═══════════════════════════════════════════════════════════════
# STEP 4 & 5: Save + Preview — orchestrated in run_cleaning_pipeline
# ═══════════════════════════════════════════════════════════════

def run_cleaning_pipeline(
    s3_raw_key: str,
    user_id: str,
    dataset_id: str,
    job_id: str,
    prompt: str,
    detected_issues: list | None = None,
) -> dict[str, Any]:
    """
    Full cleaning pipeline:
    1. Deterministic cleaning
    2. LLM plan generation (with fallback)
    3. Execute plan
    4. Save Parquet to S3 with metadata
    5. Generate preview
    """
    # ── Check cache ──
    cache_key = prompt_cache_key(dataset_id, prompt)
    try:
        r = _get_redis()
        cached = r.get(cache_key)
        if cached:
            logger.info("Cleaning pipeline cache hit", extra={"dataset_id": dataset_id})
            return json.loads(cached)
    except Exception:
        pass

    # ── Load raw data ──
    raw_bytes = download_object_to_bytes(s3_raw_key)

    if HAS_POLARS:
        if s3_raw_key.endswith((".xlsx", ".xls")):
            pdf = pd.read_excel(io.BytesIO(raw_bytes))
            df = pl.from_pandas(pdf)
        else:
            df = pl.read_csv(
                io.BytesIO(raw_bytes),
                try_parse_dates=True,
                ignore_errors=True,
                truncate_ragged_lines=True,
            )
        # Step 1: Deterministic cleaning (Polars)
        original_col_names = list(df.columns)
        df, det_steps = deterministic_clean_polars(df)
    else:
        # Pandas fallback
        if s3_raw_key.endswith((".xlsx", ".xls")):
            pdf = pd.read_excel(io.BytesIO(raw_bytes))
        else:
            pdf = pd.read_csv(io.BytesIO(raw_bytes))
        original_col_names = list(pdf.columns)
        pdf, det_steps = deterministic_clean_pandas(pdf)
        df = pl.from_pandas(pdf)

    # Build column map: cleaned → original
    col_map_cleaned = _standardize_column_names(original_col_names)
    original_column_map = {v: k for k, v in col_map_cleaned.items()}

    # Schema for LLM
    current_schema = {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)}
    sample_rows = df.head(50).to_dicts()

    # Step 2: LLM cleaning plan
    plan_result = generate_cleaning_plan(
        schema=current_schema,
        sample_rows=sample_rows,
        user_prompt=prompt,
        detected_issues=detected_issues,
    )
    plan = plan_result["plan"]
    llm_assisted = plan_result["llm_assisted"]

    # Step 3: Execute plan (if LLM provided transformations)
    all_steps = det_steps.copy()
    if plan.get("transformations"):
        df = execute_cleaning_plan(df, plan)
        for step in plan.get("steps", []):
            all_steps.append(step.get("description", "LLM transformation"))

    # Step 4: Save as Parquet with metadata
    s3_parquet_key = f"processed/{user_id}/{dataset_id}/cleaned.parquet"
    s3_csv_key = f"processed/{user_id}/{dataset_id}/cleaned.csv"

    # Convert to Arrow table and write Parquet
    arrow_table = df.to_arrow()

    parquet_meta = {
        "dataset_id": dataset_id,
        "schema_version": "1",
        "schema_fingerprint": schema_fingerprint(current_schema),
        "original_column_map": json.dumps(original_column_map),
        "cleaned_at": datetime.now(timezone.utc).isoformat(),
        "cleaning_job_id": job_id,
        "row_count": str(len(df)),
        "llm_assisted": str(llm_assisted),
        "degraded_mode": str(not llm_assisted),
    }

    parquet_buffer = io.BytesIO()
    write_parquet_with_metadata(arrow_table, parquet_buffer, parquet_meta)
    parquet_buffer.seek(0)
    upload_bytes(s3_parquet_key, parquet_buffer.read(), content_type="application/octet-stream")

    # Optional CSV copy
    csv_buffer = io.BytesIO()
    df.write_csv(csv_buffer)
    csv_buffer.seek(0)
    upload_bytes(s3_csv_key, csv_buffer.read(), content_type="text/csv")

    # Step 5: Preview
    preview_rows = df.head(settings.CLEAN_PREVIEW_ROWS).to_dicts()

    result = {
        "s3_parquet_key": s3_parquet_key,
        "s3_csv_key": s3_csv_key,
        "cleaned_row_count": len(df),
        "preview_rows": preview_rows,
        "cleaning_steps_applied": all_steps,
        "prompt_used": prompt,
        "llm_assisted": llm_assisted,
        "degraded_mode": not llm_assisted,
        "schema_fingerprint": schema_fingerprint(current_schema),
        "original_column_map": original_column_map,
        "warnings": plan_result.get("warnings", []),
        "assumptions": plan.get("assumptions", []),
    }

    # Cache result
    try:
        r = _get_redis()
        r.setex(cache_key, settings.LLM_CACHE_TTL, json.dumps(result, default=str))
    except Exception:
        pass

    logger.info(
        "Cleaning pipeline complete",
        extra={
            "dataset_id": dataset_id,
            "rows": len(df),
            "steps": len(all_steps),
            "llm_assisted": llm_assisted,
        },
    )
    return result
