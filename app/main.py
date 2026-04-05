import asyncio

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from app.api.router import api_router
from app.core.config import settings
import uvicorn

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="A general-purpose AI CSV cleaning agent using FastAPI, LangChain, and Ollama."
)

@app.on_event("startup")
async def on_startup():
    from app.db.session import engine
    from app.db.base import Base
    from app.services.ai_cleaner import warm_analysis_runtime_cache
    from app.services.object_storage import get_object_storage_service
    import app.models.chat_history
    import app.models.cleaned_data
    import app.models.analysis_suggestion
    import app.models.job
    import app.models.semantic_column_metadata
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.execute(text("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS file_url VARCHAR"))
        await conn.execute(text("ALTER TABLE cleaned_data ADD COLUMN IF NOT EXISTS source_file_id VARCHAR"))
        await conn.execute(text("ALTER TABLE cleaned_data ADD COLUMN IF NOT EXISTS cleaned_rows INTEGER"))
        await conn.execute(text("ALTER TABLE cleaned_data ADD COLUMN IF NOT EXISTS quality_score INTEGER"))
        await conn.execute(text("ALTER TABLE cleaned_data ADD COLUMN IF NOT EXISTS analysis JSON"))
        await conn.execute(text("ALTER TABLE analysis_suggestions ADD COLUMN IF NOT EXISTS source_type VARCHAR"))
        await conn.execute(text("UPDATE cleaned_data SET source_file_id = job_id WHERE source_file_id IS NULL"))
        await conn.execute(text("UPDATE cleaned_data SET cleaned_rows = 0 WHERE cleaned_rows IS NULL"))
        await conn.execute(text("UPDATE analysis_suggestions SET source_type = 'raw' WHERE source_type IS NULL"))
    await asyncio.to_thread(warm_analysis_runtime_cache)
    await asyncio.to_thread(get_object_storage_service().ensure_bucket)

# Allow CORS for potential frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the endpoints router
app.include_router(api_router, prefix="/api/v1")

@app.get("/")
def read_root():
    return {
        "message": f"Welcome to the {settings.PROJECT_NAME} API",
        "docs_url": "/docs"
    }

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
