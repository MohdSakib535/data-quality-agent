import os
import pandas as pd
import asyncio
from datetime import date, datetime

from app.services.csv_loader import iter_csv_chunks, load_csv
from app.services.ai_cleaner import (
    analyze_dataset,
    build_cleaning_chain,
    clean_dataframe_chunk,
    prompt_removes_exact_duplicates,
    prompt_requires_ai_cleaning,
)
from app.schemas.job import DatasetAnalysisResponse, CleanDataResponse
from app.core.config import settings


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


def _write_chunk_to_csv(output_path: str, df: pd.DataFrame, write_header: bool) -> None:
    mode = "w" if write_header else "a"
    df.to_csv(
        output_path,
        mode=mode,
        header=write_header,
        index=False,
        na_rep=settings.NULL_OUTPUT_TOKEN,
    )


def _clean_csv_file_with_prompt(job_id: str, prompt: str) -> tuple[str, list[dict], int]:
    cleaned_filename = f"{job_id}_cleaned.csv"
    output_path = os.path.join(settings.OUTPUT_DIR, cleaned_filename)
    preview_rows: list[dict] = []
    total_rows = 0
    wrote_header = False
    chain = build_cleaning_chain() if prompt_requires_ai_cleaning(prompt) else None
    seen_row_hashes = set() if prompt_removes_exact_duplicates(prompt) else None

    if os.path.exists(output_path):
        os.remove(output_path)

    for chunk_df in iter_csv_chunks(job_id, chunksize=settings.CHUNK_SIZE):
        cleaned_chunk = clean_dataframe_chunk(
            chunk_df,
            prompt,
            chain=chain,
            seen_row_hashes=seen_row_hashes,
        )

        preview_remaining = settings.CLEAN_PREVIEW_ROWS - len(preview_rows)
        if preview_remaining > 0 and not cleaned_chunk.empty:
            preview_rows.extend(_dataframe_to_json_records(cleaned_chunk.head(preview_remaining)))

        _write_chunk_to_csv(output_path, cleaned_chunk, write_header=not wrote_header)
        wrote_header = True
        total_rows += len(cleaned_chunk)

    if not wrote_header:
        empty_df = clean_dataframe_chunk(
            load_csv(job_id, nrows=0),
            prompt,
            chain=chain,
            seen_row_hashes=seen_row_hashes,
        )
        _write_chunk_to_csv(output_path, empty_df, write_header=True)

    return output_path, preview_rows, total_rows


async def analyze_csv(job_id: str) -> DatasetAnalysisResponse:
    """
    Load the CSV and run a pure LLM analysis over a sample asynchronously.
    """
    df = await asyncio.to_thread(load_csv, job_id)
    
    # Analyze
    response = await asyncio.to_thread(analyze_dataset, df)
    response.job_id = job_id
    
    return response

async def clean_csv_with_prompt(job_id: str, prompt: str) -> CleanDataResponse:
    """
    Load the CSV, clean it chunk-by-chunk, and write the cleaned output incrementally.
    """
    output_path, cleaned_preview, total_rows = await asyncio.to_thread(
        _clean_csv_file_with_prompt,
        job_id,
        prompt,
    )

    return CleanDataResponse(
        job_id=job_id,
        status="completed",
        cleaned_file_url=f"/api/v1/download-cleaned/{job_id}",
        cleaned_rows=total_rows,
        cleaned_data=cleaned_preview,
    )
