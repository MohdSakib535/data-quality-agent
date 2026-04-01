import csv
import os
import shutil
import uuid
from functools import lru_cache

from fastapi import UploadFile
import pandas as pd

from app.core.config import settings

SUPPORTED_UPLOAD_EXTENSIONS = {".csv", ".xlsx", ".xls"}


# Prefer faster parsers when available; fall back to pandas defaults.
_CSV_ENGINES: tuple[str, ...] = ("pyarrow", "c", "python")


def _get_upload_extension(filename: str | None) -> str:
    if not filename:
        return ""
    return os.path.splitext(filename)[1].lower()


def is_supported_upload_file(upload_file: UploadFile) -> bool:
    return _get_upload_extension(upload_file.filename) in SUPPORTED_UPLOAD_EXTENSIONS


def _save_csv_upload(upload_file: UploadFile, file_path: str) -> None:
    upload_file.file.seek(0)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)


def _save_excel_upload(upload_file: UploadFile, file_path: str, extension: str) -> None:
    upload_file.file.seek(0)
    try:
        engine = "openpyxl" if extension == ".xlsx" else "xlrd"
        dataframe = pd.read_excel(upload_file.file, dtype=object, engine=engine)
    except ImportError as exc:
        raise ValueError(
            "Excel support requires optional dependencies: openpyxl (.xlsx) and xlrd (.xls)."
        ) from exc
    except Exception as exc:
        raise ValueError(
            "Unable to parse the Excel file. Please upload a valid .xlsx or .xls file."
        ) from exc

    dataframe.to_csv(file_path, index=False)


def save_upload_file(upload_file: UploadFile) -> str:
    """Save the uploaded file and return a job_id."""
    extension = _get_upload_extension(upload_file.filename)
    if extension not in SUPPORTED_UPLOAD_EXTENSIONS:
        raise ValueError("Only CSV or Excel files (.csv, .xlsx, .xls) are allowed.")

    job_id = str(uuid.uuid4())
    file_path = os.path.join(settings.UPLOAD_DIR, f"{job_id}.csv")

    if extension == ".csv":
        _save_csv_upload(upload_file, file_path)
    else:
        _save_excel_upload(upload_file, file_path, extension)

    return job_id


def _get_file_path(job_id: str) -> str:
    file_path = os.path.join(settings.UPLOAD_DIR, f"{job_id}.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File for job {job_id} not found.")
    return file_path


@lru_cache(maxsize=256)
def _detect_csv_options(file_path: str) -> dict[str, str]:
    """Detect CSV encoding and delimiter with a small sample. Cached per file."""
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
    """
    Read CSV using the fastest available engine (pyarrow -> C -> python).
    Falls back to alternate encodings if the first attempt fails.
    """
    base_kwargs = {
        "sep": delimiter,
        "encoding": encoding,
        "memory_map": True,
        **kwargs,
    }

    encodings_to_try = [encoding]
    if "latin1" not in encodings_to_try:
        encodings_to_try.append("latin1")
    if "utf-8-sig" not in encodings_to_try:
        encodings_to_try.append("utf-8-sig")

    for enc in encodings_to_try:
        for engine in _CSV_ENGINES:
            try:
                return pd.read_csv(file_path, engine=engine, **{**base_kwargs, "encoding": enc})
            except UnicodeDecodeError:
                # Try next encoding
                break
            except (ImportError, ValueError, TypeError):
                # Engine not installed or unsupported; try next one
                continue

    # Final fallback: let pandas decide.
    return pd.read_csv(file_path, sep=delimiter, **kwargs)


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
