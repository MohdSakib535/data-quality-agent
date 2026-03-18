import os
import pandas as pd
from app.services.csv_loader import load_csv
from app.services.ai_cleaner import analyze_dataset, clean_dataset_with_prompt
from app.schemas.models import DatasetAnalysisResponse, CleanDataResponse
from app.core.config import settings


def _dataframe_to_json_records(df: pd.DataFrame):
    cleaned_df = df.where(pd.notnull(df), None)
    return cleaned_df.to_dict(orient="records")

def analyze_csv(job_id: str) -> DatasetAnalysisResponse:
    """
    Load the CSV and run a pure LLM analysis over a sample.
    """
    df = load_csv(job_id)
    
    # Analyze
    response = analyze_dataset(df)
    response.job_id = job_id
    
    return response

def clean_csv_with_prompt(job_id: str, prompt: str) -> CleanDataResponse:
    """
    Load the CSV, clean it using the provided natural language prompt in batches,
    and save the output.
    """
    df = load_csv(job_id)
    
    cleaned_df = clean_dataset_with_prompt(df, prompt)
    
    # Save the cleaned dataframe
    cleaned_filename = f"{job_id}_cleaned.csv"
    output_path = os.path.join(settings.OUTPUT_DIR, cleaned_filename)
    cleaned_df.to_csv(output_path, index=False)
    
    return CleanDataResponse(
        job_id=job_id,
        status="completed",
        cleaned_file_url=f"/api/v1/download-cleaned/{job_id}",
        cleaned_data=_dataframe_to_json_records(cleaned_df),
    )
