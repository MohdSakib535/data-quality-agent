"""
Dataset analysis service.
Deterministic profiling + LLM-assisted quality scoring and suggestions.
"""
import io
import json
import logging
from typing import Any

import polars as pl

from app.services.llm import call_llm_with_fallback
from app.services.storage import download_object_to_bytes, stream_object
from app.core.config import settings

logger = logging.getLogger(__name__)


def _load_sample_dataframe(s3_key: str, max_rows: int | None = None) -> pl.DataFrame:
    """
    Download raw file from S3 and load into a Polars DataFrame.
    Supports CSV and XLSX. Limits to max_rows for analysis sampling.
    """
    max_rows = max_rows or settings.ANALYSIS_SAMPLE_ROWS
    raw_bytes = download_object_to_bytes(s3_key)

    if s3_key.endswith((".xlsx", ".xls")):
        import pandas as pd
        pdf = pd.read_excel(io.BytesIO(raw_bytes), nrows=max_rows)
        df = pl.from_pandas(pdf)
    else:
        # CSV
        df = pl.read_csv(
            io.BytesIO(raw_bytes),
            n_rows=max_rows,
            try_parse_dates=True,
            ignore_errors=True,
            truncate_ragged_lines=True,
        )

    logger.info(
        "Loaded sample dataframe",
        extra={"s3_key": s3_key, "rows": len(df), "cols": len(df.columns)},
    )
    return df


def _count_total_rows(s3_key: str) -> int:
    """Count total rows without loading entire file into memory."""
    raw_bytes = download_object_to_bytes(s3_key)
    if s3_key.endswith((".xlsx", ".xls")):
        import pandas as pd
        pdf = pd.read_excel(io.BytesIO(raw_bytes))
        return len(pdf)
    else:
        # Count newlines for CSV (subtract 1 for header)
        count = raw_bytes.count(b"\n")
        return max(0, count - 1)


def deterministic_profile(df: pl.DataFrame, total_rows: int | None = None) -> dict[str, Any]:
    """
    Run deterministic profiling on a Polars DataFrame.
    Returns profiling dict with schema, null stats, duplicates, etc.
    """
    row_count = total_rows if total_rows is not None else len(df)
    column_count = len(df.columns)

    # Schema snapshot
    schema_snapshot = {
        col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)
    }

    # Null percentage per column
    null_stats = {}
    for col in df.columns:
        null_count = df[col].null_count()
        null_pct = round((null_count / len(df)) * 100, 2) if len(df) > 0 else 0.0
        null_stats[col] = {
            "null_count": int(null_count),
            "null_percentage": null_pct,
        }

    # Duplicate row count
    duplicate_count = int(len(df) - len(df.unique()))

    # Unique value counts for low-cardinality columns (< 50 unique values)
    unique_value_analysis = {}
    for col in df.columns:
        try:
            n_unique = df[col].n_unique()
            if n_unique <= 50:
                value_counts = df[col].value_counts().sort("count", descending=True)
                unique_value_analysis[col] = {
                    "n_unique": int(n_unique),
                    "top_values": [
                        {"value": str(row[col]), "count": int(row["count"])}
                        for row in value_counts.head(10).to_dicts()
                    ],
                }
        except Exception:
            continue

    # Date format detection
    date_columns = []
    for col in df.columns:
        dtype_str = str(df[col].dtype).lower()
        if "date" in dtype_str or "time" in dtype_str:
            date_columns.append(col)

    profile = {
        "row_count": row_count,
        "column_count": column_count,
        "schema_snapshot": schema_snapshot,
        "null_stats": null_stats,
        "duplicate_count": duplicate_count,
        "unique_value_analysis": unique_value_analysis,
        "date_columns": date_columns,
        "sample_row_count": len(df),
    }

    logger.info(
        "Deterministic profiling complete",
        extra={"rows": row_count, "cols": column_count, "duplicates": duplicate_count},
    )
    return profile


def llm_analyze(
    schema_snapshot: dict,
    sample_rows: list[dict],
    null_stats: dict,
    duplicate_count: int,
) -> dict[str, Any]:
    """
    Call LLM to analyze dataset quality and suggest improvements.
    Uses call_llm_with_fallback — never calls LLM directly.
    """
    prompt = f"""Analyze this dataset and provide quality assessment.

SCHEMA:
{json.dumps(schema_snapshot, indent=2)}

SAMPLE DATA (first 50 rows):
{json.dumps(sample_rows[:50], indent=2, default=str)}

NULL STATISTICS:
{json.dumps(null_stats, indent=2)}

DUPLICATE ROWS: {duplicate_count}

Return ONLY valid JSON with this exact structure:
{{
    "suggestions": [
        {{
            "column": "column_name or null for general suggestions",
            "issue": "description of the issue",
            "severity": "high|medium|low",
            "recommendation": "what to do"
        }}
    ],
    "quality_score": <integer 0-100>,
    "issues": [
        {{
            "type": "missing_data|duplicates|inconsistent_format|outliers|other",
            "description": "...",
            "affected_columns": ["..."]
        }}
    ]
}}"""

    fallback = {
        "suggestions": [],
        "quality_score": None,
        "issues": [],
    }

    result = call_llm_with_fallback(
        prompt=prompt,
        expected_keys=["suggestions", "quality_score", "issues"],
        context="analysis",
        fallback_response=fallback,
    )

    return {
        "llm_data": result.data,
        "llm_assisted": result.llm_assisted,
        "tier_used": result.tier_used,
        "warnings": result.warnings,
    }


def run_full_analysis(s3_key: str) -> dict[str, Any]:
    """
    Complete analysis pipeline:
    1. Load sample from S3
    2. Deterministic profiling
    3. LLM analysis (with fallback)
    4. Merge results
    """
    # Step 1: Load sample
    df = _load_sample_dataframe(s3_key)
    total_rows = _count_total_rows(s3_key)

    # Step 2: Deterministic profiling
    profile = deterministic_profile(df, total_rows=total_rows)

    # Step 3: LLM analysis
    sample_rows = df.head(50).to_dicts()
    llm_result = llm_analyze(
        schema_snapshot=profile["schema_snapshot"],
        sample_rows=sample_rows,
        null_stats=profile["null_stats"],
        duplicate_count=profile["duplicate_count"],
    )

    # Step 4: Merge
    quality_score = None
    llm_suggestions = []
    if llm_result["llm_assisted"] and isinstance(llm_result["llm_data"], dict):
        quality_score = llm_result["llm_data"].get("quality_score")
        llm_suggestions = llm_result["llm_data"].get("suggestions", [])

    merged = {
        "row_count": profile["row_count"],
        "column_count": profile["column_count"],
        "schema_snapshot": profile["schema_snapshot"],
        "quality_score": quality_score,
        "null_stats": profile["null_stats"],
        "duplicate_count": profile["duplicate_count"],
        "llm_suggestions": llm_suggestions,
        "llm_assisted": llm_result["llm_assisted"],
        "tier_used": llm_result["tier_used"],
        "warnings": llm_result["warnings"],
    }

    logger.info("Full analysis complete", extra={"llm_assisted": llm_result["llm_assisted"]})
    return merged
