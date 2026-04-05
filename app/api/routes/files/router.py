from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models.job import Job
from app.schemas.job import FileUploadResponse
from app.services.csv_loader import (
    SUPPORTED_UPLOAD_EXTENSIONS,
    is_supported_upload_file,
    save_upload_file,
)

router = APIRouter(prefix="/files", tags=["files"])


@router.post("/upload", response_model=FileUploadResponse)
async def upload_file(file: UploadFile = File(...), db: AsyncSession = Depends(get_db)):
    """Upload a CSV/Excel file, store it in object storage, and persist the file URL."""
    if not is_supported_upload_file(file):
        supported_extensions = ", ".join(sorted(SUPPORTED_UPLOAD_EXTENSIONS))
        raise HTTPException(
            status_code=400,
            detail=f"Only CSV or Excel files are allowed ({supported_extensions}).",
        )

    try:
        file_id, file_url = save_upload_file(file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    db_job = Job(
        id=file_id,
        status="uploaded",
        filename=file.filename,
        file_url=file_url,
    )
    db.add(db_job)
    await db.commit()

    return FileUploadResponse(
        file_id=file_id,
        filename=file.filename,
        file_url=file_url,
        status="uploaded",
    )
