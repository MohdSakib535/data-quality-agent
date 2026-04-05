import uuid
import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.error_utils import raise_http_for_service_error
from app.db.session import get_db
from app.models.analysis_suggestion import AnalysisSuggestion
from app.models.cleaned_data import CleanedData
from app.models.job import Job
from app.schemas.job import DataSuggestion, DatasetAnalysisResponse, SuggestionDetailResponse
from app.services.pipeline import analyze_csv

router = APIRouter(prefix="/analysis", tags=["analysis"])
logger = logging.getLogger(__name__)


@router.post("/{file_id}", response_model=DatasetAnalysisResponse)
async def analyze_uploaded_file(
    file_id: str,
    is_clean: bool = Query(default=False, description="Analyze the cleaned file when true; raw upload otherwise."),
    db: AsyncSession = Depends(get_db),
):
    """Analyze a previously uploaded CSV/Excel file by file ID."""
    db_job = await db.get(Job, file_id)
    if db_job is None:
        raise HTTPException(status_code=404, detail="File ID not found.")

    cleaned_result = None
    source_type = "clean" if is_clean else "raw"
    if is_clean:
        cleaned_result = await db.get(CleanedData, file_id)
        if cleaned_result is None:
            raise HTTPException(status_code=404, detail="Cleaned data not found. Run cleaning first.")

    db_job.status = "analyzing"
    db_job.message = None
    await db.commit()

    try:
        analysis_response = await analyze_csv(file_id, is_clean=is_clean)
        await db.execute(
            delete(AnalysisSuggestion).where(
                AnalysisSuggestion.job_id == file_id,
                AnalysisSuggestion.source_type == source_type,
            )
        )

        suggestion_rows = []
        response_suggestions = []
        for suggestion in analysis_response.suggestions:
            suggestion_id = str(uuid.uuid4())
            suggestion_row = AnalysisSuggestion(
                id=suggestion_id,
                job_id=file_id,
                source_type=source_type,
                issue_description=suggestion.issue_description,
                priority=suggestion.priority,
                resolution_prompt=suggestion.resolution_prompt,
            )
            suggestion_rows.append(suggestion_row)
            response_suggestions.append(
                DataSuggestion(
                    id=suggestion_id,
                    issue_description=suggestion.issue_description,
                    priority=suggestion.priority,
                    resolution_prompt=suggestion.resolution_prompt,
                )
            )

        db.add_all(suggestion_rows)
        analysis_response.suggestions = response_suggestions
        db_job.status = "analyzed"
        if is_clean:
            cleaned_result.analysis = analysis_response.model_dump()
            cleaned_result.quality_score = analysis_response.quality_score
        else:
            db_job.analysis = analysis_response.model_dump()
            db_job.quality_score = analysis_response.quality_score
        await db.commit()
        return analysis_response
    except Exception as exc:
        await db.rollback()
        db_job.status = "failed"
        db_job.message = str(exc)
        await db.commit()
        raise_http_for_service_error(exc, operation="Dataset analysis", logger=logger)


@router.get("/suggestions/{suggestion_id}", response_model=SuggestionDetailResponse)
async def get_suggestion_detail(
    suggestion_id: str,
    db: AsyncSession = Depends(get_db),
):
    suggestion = await db.get(AnalysisSuggestion, suggestion_id)
    if suggestion is None:
        raise HTTPException(status_code=404, detail="Suggestion ID not found.")
    return suggestion
