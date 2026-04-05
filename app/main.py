import asyncio

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from app.api.routes import router
from app.routers.chat_with_data import router as chat_router
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
    import app.models.job
    import app.models.semantic_column_metadata
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.execute(text("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS file_url VARCHAR"))
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
app.include_router(router, prefix="/api/v1")
app.include_router(chat_router)

@app.get("/")
def read_root():
    return {
        "message": f"Welcome to the {settings.PROJECT_NAME} API",
        "docs_url": "/docs"
    }

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
