import csv
import os
import shutil
import uuid
from fastapi import UploadFile
import pandas as pd
from app.core.config import settings

def save_upload_file(upload_file: UploadFile) -> str:
    """Save the uploaded file and return a job_id."""
    job_id = str(uuid.uuid4())
    file_path = os.path.join(settings.UPLOAD_DIR, f"{job_id}.csv")
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
        
    return job_id

def load_csv(job_id: str) -> pd.DataFrame:
    """Load the saved CSV file into a pandas DataFrame."""
    file_path = os.path.join(settings.UPLOAD_DIR, f"{job_id}.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File for job {job_id} not found.")

    with open(file_path, "rb") as raw_file:
        raw_sample = raw_file.read(4096)

    encoding = "utf-8-sig"
    sample = raw_sample.decode(encoding, errors="ignore")

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = ","

    try:
        df = pd.read_csv(file_path, encoding=encoding, sep=delimiter)
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding="latin1", sep=delimiter)

    return df
