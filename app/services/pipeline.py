import os
import pandas as pd
import asyncio
from datetime import date, datetime

from app.services.csv_loader import load_csv
from app.services.ai_cleaner import analyze_dataset, clean_dataset_with_prompt
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
    Load the CSV, clean it using the provided natural language prompt in batches,
    and save the output asynchronously.
    """
    df = await asyncio.to_thread(load_csv, job_id)
    
    cleaned_df = await asyncio.to_thread(clean_dataset_with_prompt, df, prompt)

    cleaned_records = _dataframe_to_json_records(cleaned_df)

    # Save the cleaned dataframe
    cleaned_filename = f"{job_id}_cleaned.csv"
    output_path = os.path.join(settings.OUTPUT_DIR, cleaned_filename)
    await asyncio.to_thread(
        cleaned_df.to_csv,
        output_path,
        index=False,
        na_rep=settings.NULL_OUTPUT_TOKEN,
    )

    return CleanDataResponse(
        job_id=job_id,
        status="completed",
        cleaned_file_url=f"/api/v1/download-cleaned/{job_id}",
        cleaned_data=cleaned_records,
    )
