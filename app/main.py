import asyncio

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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
    from app.db.base import Base
    from app.db.session import engine
    from app.services.ai_cleaner import warm_analysis_runtime_cache
    from app.services.object_storage import get_object_storage_service
    import app.models.analysis_suggestion  # noqa: F401
    import app.models.chat_history  # noqa: F401
    import app.models.cleaned_data  # noqa: F401
    import app.models.job  # noqa: F401
    import app.models.semantic_column_metadata  # noqa: F401

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

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
