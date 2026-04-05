import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models.analysis_suggestion import AnalysisSuggestion
from app.models.job import Job
from app.schemas.job import DataSuggestion, DatasetAnalysisResponse, SuggestionDetailResponse
from app.services.pipeline import analyze_csv

router = APIRouter(prefix="/analysis", tags=["analysis"])


@router.post("/{file_id}", response_model=DatasetAnalysisResponse)
async def analyze_uploaded_file(file_id: str, db: AsyncSession = Depends(get_db)):
    """Analyze a previously uploaded CSV/Excel file by file ID."""
    db_job = await db.get(Job, file_id)
    if db_job is None:
        raise HTTPException(status_code=404, detail="File ID not found.")

    db_job.status = "analyzing"
    db_job.message = None
    await db.commit()

    try:
        analysis_response = await analyze_csv(file_id)
        await db.execute(
            delete(AnalysisSuggestion).where(AnalysisSuggestion.job_id == file_id)
        )

        suggestion_rows = []
        response_suggestions = []
        for suggestion in analysis_response.suggestions:
            suggestion_id = str(uuid.uuid4())
            suggestion_row = AnalysisSuggestion(
                id=suggestion_id,
                job_id=file_id,
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
        db_job.analysis = analysis_response.model_dump()
        db_job.quality_score = analysis_response.quality_score
        await db.commit()
        return analysis_response
    except FileNotFoundError as exc:
        await db.rollback()
        db_job.status = "failed"
        db_job.message = str(exc)
        await db.commit()
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        await db.rollback()
        db_job.status = "failed"
        db_job.message = str(exc)
        await db.commit()
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/suggestions/{suggestion_id}", response_model=SuggestionDetailResponse)
async def get_suggestion_detail(
    suggestion_id: str,
    db: AsyncSession = Depends(get_db),
):
    suggestion = await db.get(AnalysisSuggestion, suggestion_id)
    if suggestion is None:
        raise HTTPException(status_code=404, detail="Suggestion ID not found.")
    return suggestion
