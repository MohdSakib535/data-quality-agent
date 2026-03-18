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


def _get_file_path(job_id: str) -> str:
    file_path = os.path.join(settings.UPLOAD_DIR, f"{job_id}.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File for job {job_id} not found.")
    return file_path


def _detect_csv_options(file_path: str) -> dict[str, str]:
    """Detect CSV encoding and delimiter with a small sample."""
    with open(file_path, "rb") as raw_file:
        raw_sample = raw_file.read(4096)

    detected_encoding = "utf-8-sig"
    try:
        sample = raw_sample.decode(detected_encoding)
    except UnicodeDecodeError:
        detected_encoding = "latin1"
        sample = raw_sample.decode(detected_encoding, errors="ignore")

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = ","

    return {"encoding": detected_encoding, "delimiter": delimiter}


def _read_csv(file_path: str, encoding: str, delimiter: str, **kwargs):
    try:
        return pd.read_csv(file_path, encoding=encoding, sep=delimiter, **kwargs)
    except UnicodeDecodeError:
        fallback_encoding = "latin1" if encoding != "latin1" else "utf-8-sig"
        return pd.read_csv(file_path, encoding=fallback_encoding, sep=delimiter, **kwargs)


def load_csv(job_id: str, **kwargs) -> pd.DataFrame:
    """Load the saved CSV file into a pandas DataFrame."""
    file_path = _get_file_path(job_id)
    csv_options = _detect_csv_options(file_path)
    return _read_csv(file_path, csv_options["encoding"], csv_options["delimiter"], **kwargs)


def iter_csv_chunks(job_id: str, chunksize: int | None = None):
    """Yield CSV data in chunks so large files don't need to be loaded at once."""
    file_path = _get_file_path(job_id)
    csv_options = _detect_csv_options(file_path)
    effective_chunksize = chunksize or settings.CHUNK_SIZE
    return _read_csv(
        file_path,
        csv_options["encoding"],
        csv_options["delimiter"],
        chunksize=effective_chunksize,
    )
