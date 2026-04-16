import os
import asyncio
import logging
import tempfile
import time
from datetime import date, datetime

import pandas as pd

from app.services.csv_loader import (
    count_csv_rows,
    iter_cleaned_csv_chunks,
    iter_csv_chunks,
    load_cleaned_csv,
    load_csv,
)
from app.services.ai_cleaner import (
    AICleaningPlan,
    analyze_dataset,
    build_cleaning_chain,
    clean_dataframe_chunk,
    cleaning_strategy_uses_llm,
    plan_ai_cleaning,
    prompt_has_deterministic_cleaning_steps,
    prompt_removes_exact_duplicates,
)
from app.services.deterministic_cleaner import build_dataset_profile_from_chunks
from app.services.object_storage import (
    cleaned_output_key,
    get_object_storage_service,
)
from app.schemas.job import DatasetAnalysisResponse, CleanDataResponse
from app.core.config import settings

logger = logging.getLogger(__name__)


def _sanitize_json_value(value):
    if isinstance(value, dict):
        return {key: _sanitize_json_value(item) for key, item in value.items()}

    if isinstance(value, list):
        return [_sanitize_json_value(item) for item in value]

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if pd.isna(value):
        return settings.NULL_OUTPUT_TOKEN

    if hasattr(value, "item"):
        return _sanitize_json_value(value.item())

    return value


def _dataframe_to_json_records(df: pd.DataFrame):
    cleaned_df = df.astype(object).where(pd.notnull(df), settings.NULL_OUTPUT_TOKEN)
    return _sanitize_json_value(cleaned_df.to_dict(orient="records"))


def _dataframes_match(left_df: pd.DataFrame, right_df: pd.DataFrame) -> bool:
    if list(left_df.columns) != list(right_df.columns):
        return False
    if len(left_df.index) != len(right_df.index):
        return False

    left_values = left_df.astype(object).where(pd.notnull(left_df), settings.NULL_OUTPUT_TOKEN).astype(str)
    right_values = right_df.astype(object).where(pd.notnull(right_df), settings.NULL_OUTPUT_TOKEN).astype(str)
    return left_values.equals(right_values)


def _build_clean_message(
    *,
    source_type: str,
    strategy: str,
    target_columns: list[str],
    cleaned_rows: int,
    preview_rows_returned: int,
    preview_limited: bool,
    changes_detected: bool,
) -> str:
    source_label = "original upload" if source_type == "raw" else "previous cleaned output"
    target_label = ", ".join(target_columns) if target_columns else "all columns"
    strategy_label = (
        "deterministic (privacy-safe fallback)"
        if strategy == "deterministic_privacy_fallback"
        else strategy
    )
    preview_note = (
        f"API response includes the first {preview_rows_returned} row(s) only"
        if preview_limited
        else f"API response includes {preview_rows_returned} row(s)"
    )
    if changes_detected:
        status_note = "Changes were detected."
    else:
        status_note = "Prompt was processed, but no data changes were detected."

    privacy_note = ""
    if strategy == "deterministic_privacy_fallback":
        privacy_note = (
            " AI cleaning was blocked for privacy-sensitive columns, so only deterministic cleanup steps were applied."
        )

    return (
        f"{status_note} Source: {source_label}. Strategy: {strategy_label}. "
        f"Target columns: {target_label}. {preview_note} out of {cleaned_rows} cleaned row(s)."
        f"{privacy_note}"
    )


def _build_analysis_profile(job_id: str, is_clean: bool = False) -> dict:
    """
    Build a deterministic profile from the full dataset in chunks so the quality
    score is stable across repeated clean/analyze cycles.
    """
    chunk_loader = iter_cleaned_csv_chunks if is_clean else iter_csv_chunks
    analysis_chunksize = max(settings.CHUNK_SIZE, settings.CHUNK_SIZE * 10)
    return build_dataset_profile_from_chunks(
        chunk_loader(job_id, chunksize=analysis_chunksize)
    )


def _load_cleaning_planner_sample(job_id: str, use_cleaned_source: bool = False) -> pd.DataFrame:
    sample_rows = max(1, settings.AI_PLANNER_SAMPLE_ROWS)
    loader = load_cleaned_csv if use_cleaned_source else load_csv
    return loader(job_id, nrows=sample_rows)


def _clean_csv_file_with_prompt(
    job_id: str,
    prompt: str,
    use_cleaned_source: bool = False,
) -> tuple[str, list[dict], int, str, list[str], bool]:
    preview_rows: list[dict] = []
    total_rows = 0
    wrote_header = False
    changes_detected = False
    planner_sample_df = _load_cleaning_planner_sample(job_id, use_cleaned_source)
    cleaning_plan = plan_ai_cleaning(planner_sample_df, prompt)
    if cleaning_plan.strategy == "row_level_ai":
        row_limit = settings.AI_ROW_LEVEL_LARGE_DATASET_THRESHOLD + 1
        total_rows_hint = count_csv_rows(job_id, is_clean=use_cleaned_source, limit=row_limit)
        cleaning_plan = plan_ai_cleaning(
            planner_sample_df,
            prompt,
            total_rows=total_rows_hint,
        )
        if cleaning_plan.strategy == "unsupported_large_ai":
            raise ValueError(cleaning_plan.reason or "Unsupported large-dataset AI cleaning prompt.")
    if cleaning_plan.strategy == "privacy_blocked":
        if prompt_has_deterministic_cleaning_steps(prompt):
            logger.info(
                "[cleaning] job_id=%s privacy-blocked AI plan fell back to deterministic cleanup. target_columns=%s",
                job_id,
                cleaning_plan.target_columns,
            )
            cleaning_plan = AICleaningPlan(
                strategy="deterministic_privacy_fallback",
                target_columns=cleaning_plan.target_columns,
                reason=cleaning_plan.reason,
            )
        else:
            raise ValueError(cleaning_plan.reason or "AI cleaning is blocked for the requested columns.")

    requires_ai_cleaning = cleaning_plan.requires_ai
    logger.info(
        "[cleaning] job_id=%s strategy=%s target_columns=%s",
        job_id,
        cleaning_plan.strategy,
        cleaning_plan.target_columns,
    )
    chain = build_cleaning_chain() if requires_ai_cleaning else None
    ai_row_cache = {} if requires_ai_cleaning else None
    seen_row_hashes = set() if prompt_removes_exact_duplicates(prompt) else None
    effective_chunksize = (
        settings.CHUNK_SIZE
        if requires_ai_cleaning
        else max(settings.CHUNK_SIZE, settings.CHUNK_SIZE * 10)
    )
    chunk_loader = iter_cleaned_csv_chunks if use_cleaned_source else iter_csv_chunks
    empty_loader = load_cleaned_csv if use_cleaned_source else load_csv

    fd, output_path = tempfile.mkstemp(prefix=f"{job_id}_cleaned_", suffix=".csv")
    os.close(fd)

    try:
        with open(output_path, "w", encoding="utf-8", newline="") as output_file:
            for chunk_df in chunk_loader(job_id, chunksize=effective_chunksize):
                cleaned_chunk = clean_dataframe_chunk(
                    chunk_df,
                    prompt,
                    chain=chain,
                    plan=cleaning_plan,
                    seen_row_hashes=seen_row_hashes,
                    ai_row_cache=ai_row_cache,
                )
                if not changes_detected and not _dataframes_match(chunk_df, cleaned_chunk):
                    changes_detected = True

                preview_remaining = settings.CLEAN_PREVIEW_ROWS - len(preview_rows)
                if preview_remaining > 0 and not cleaned_chunk.empty:
                    preview_rows.extend(_dataframe_to_json_records(cleaned_chunk.head(preview_remaining)))

                cleaned_chunk.to_csv(
                    output_file,
                    header=not wrote_header,
                    index=False,
                    na_rep=settings.NULL_OUTPUT_TOKEN,
                )
                wrote_header = True
                total_rows += len(cleaned_chunk)

            if not wrote_header:
                source_empty_df = empty_loader(job_id, nrows=0)
                empty_df = clean_dataframe_chunk(
                    source_empty_df,
                    prompt,
                    chain=chain,
                    plan=cleaning_plan,
                    seen_row_hashes=seen_row_hashes,
                    ai_row_cache=ai_row_cache,
                )
                if not changes_detected and not _dataframes_match(source_empty_df, empty_df):
                    changes_detected = True
                empty_df.to_csv(
                    output_file,
                    header=True,
                    index=False,
                    na_rep=settings.NULL_OUTPUT_TOKEN,
                )

        file_url = get_object_storage_service().upload_file(output_path, cleaned_output_key(job_id))
        return (
            file_url,
            preview_rows,
            total_rows,
            cleaning_plan.strategy,
            cleaning_plan.target_columns,
            changes_detected,
        )
    finally:
        if os.path.exists(output_path):
            os.remove(output_path)


async def analyze_csv(job_id: str, is_clean: bool = False) -> DatasetAnalysisResponse:
    """
    Run full-file deterministic analysis in chunks so quality scoring is stable.
    """
    total_start = time.perf_counter()
    profile_build_ms = 0.0
    llm_request_ms = 0.0

    try:
        profile_build_start = time.perf_counter()
        profile = await asyncio.to_thread(_build_analysis_profile, job_id, is_clean)
        profile_build_ms = (time.perf_counter() - profile_build_start) * 1000

        response, llm_request_ms = await analyze_dataset(profile)
        response.job_id = job_id
        response.source_type = "clean" if is_clean else "raw"
        return response
    finally:
        total_analysis_ms = (time.perf_counter() - total_start) * 1000
        logger.info("[analysis] profile_build_ms=%d", round(profile_build_ms))
        logger.info("[analysis] llm_request_ms=%d", round(llm_request_ms))
        logger.info("[analysis] total_analysis_ms=%d", round(total_analysis_ms))

async def clean_csv_with_prompt(
    job_id: str,
    prompt: str,
    use_cleaned_source: bool = False,
) -> CleanDataResponse:
    """
    Load the CSV, clean it chunk-by-chunk, and write the cleaned output incrementally.
    """
    source_type = "cleaned" if use_cleaned_source else "raw"
    try:
        _, cleaned_preview, total_rows, cleaning_strategy, target_columns, changes_detected = await asyncio.to_thread(
            _clean_csv_file_with_prompt,
            job_id,
            prompt,
            use_cleaned_source,
        )
    except FileNotFoundError:
        if not use_cleaned_source:
            raise

        logger.warning(
            "Cleaned source for job_id=%s not found, falling back to raw upload for cleaning.",
            job_id,
        )
        source_type = "raw"
        _, cleaned_preview, total_rows, cleaning_strategy, target_columns, changes_detected = await asyncio.to_thread(
            _clean_csv_file_with_prompt,
            job_id,
            prompt,
            False,
        )
    preview_rows_returned = len(cleaned_preview)
    preview_limited = total_rows > preview_rows_returned
    message = _build_clean_message(
        source_type=source_type,
        strategy=cleaning_strategy,
        target_columns=target_columns,
        cleaned_rows=total_rows,
        preview_rows_returned=preview_rows_returned,
        preview_limited=preview_limited,
        changes_detected=changes_detected,
    )

    return CleanDataResponse(
        job_id=job_id,
        source_file_id=job_id,
        status="completed",
        prompt=prompt,
        source_type=source_type,
        cleaning_strategy=cleaning_strategy,
        llm_used=cleaning_strategy_uses_llm(cleaning_strategy),
        target_columns=target_columns,
        cleaned_file_url=f"/api/v1/clean/{job_id}/download",
        message=message,
        cleaned_rows=total_rows,
        preview_rows_returned=preview_rows_returned,
        preview_limited=preview_limited,
        changes_detected=changes_detected,
        cleaned_data=cleaned_preview,
    )
