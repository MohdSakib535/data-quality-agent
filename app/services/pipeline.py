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


def _load_analysis_sample(job_id: str) -> pd.DataFrame:
    """
    Load only a bounded number of rows for upload-time analysis so large files
    return fast without scanning the full dataset.
    """
    max_rows = max(1, settings.ANALYSIS_SAMPLE_ROWS)
    sample_chunks: list[pd.DataFrame] = []
    collected_rows = 0
    chunk_size = min(max_rows, max(settings.CHUNK_SIZE, 1000))

    for chunk_df in iter_csv_chunks(job_id, chunksize=chunk_size):
        sample_chunks.append(chunk_df)
        collected_rows += len(chunk_df)
        if collected_rows >= max_rows:
            break

    if not sample_chunks:
        return load_csv(job_id, nrows=0)

    sample_df = pd.concat(sample_chunks, ignore_index=True)
    if len(sample_df) > max_rows:
        return sample_df.head(max_rows)
    return sample_df


def _clean_csv_file_with_prompt(job_id: str, prompt: str) -> tuple[str, list[dict], int]:
    cleaned_filename = f"{job_id}_cleaned.csv"
    output_path = os.path.join(settings.OUTPUT_DIR, cleaned_filename)
    preview_rows: list[dict] = []
    total_rows = 0
    wrote_header = False
    requires_ai_cleaning = prompt_requires_ai_cleaning(prompt)
    chain = build_cleaning_chain() if requires_ai_cleaning else None
    ai_row_cache = {} if requires_ai_cleaning else None
    seen_row_hashes = set() if prompt_removes_exact_duplicates(prompt) else None
    effective_chunksize = (
        settings.CHUNK_SIZE
        if requires_ai_cleaning
        else max(settings.CHUNK_SIZE, settings.CHUNK_SIZE * 10)
    )

    if os.path.exists(output_path):
        os.remove(output_path)

    with open(output_path, "w", encoding="utf-8", newline="") as output_file:
        for chunk_df in iter_csv_chunks(job_id, chunksize=effective_chunksize):
            cleaned_chunk = clean_dataframe_chunk(
                chunk_df,
                prompt,
                chain=chain,
                seen_row_hashes=seen_row_hashes,
                ai_row_cache=ai_row_cache,
            )

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
            empty_df = clean_dataframe_chunk(
                load_csv(job_id, nrows=0),
                prompt,
                chain=chain,
                seen_row_hashes=seen_row_hashes,
                ai_row_cache=ai_row_cache,
            )
            empty_df.to_csv(
                output_file,
                header=True,
                index=False,
                na_rep=settings.NULL_OUTPUT_TOKEN,
            )

    return output_path, preview_rows, total_rows


async def analyze_csv(job_id: str) -> DatasetAnalysisResponse:
    """
    Run analysis over a bounded sample for fast upload response on large files.
    """
    df = await asyncio.to_thread(_load_analysis_sample, job_id)
    
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
