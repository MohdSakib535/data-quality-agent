"""
FastAPI application factory.
Registers middleware, routers, and startup/shutdown hooks.
"""
import logging
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api.middleware import setup_middleware
from app.api.routers import datasets, jobs, queries
from app.db.session import async_engine
from app.db.base import Base
import app.models.dataset  # noqa: F401
import app.models.query  # noqa: F401

# ── Structured logging ───────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","message":"%(message)s"}',
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    application = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description="Production-grade dataset intelligence platform: upload, analyze, clean, and query datasets using NL→SQL.",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Custom middleware
    setup_middleware(application)

    # Routers
    application.include_router(datasets.router)
    application.include_router(jobs.router)
    application.include_router(queries.router)

    @application.on_event("startup")
    async def on_startup():
        # Auto-create tables on startup (idempotent)
        async with async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Application starting up", extra={"version": settings.VERSION})

    @application.on_event("shutdown")
    async def on_shutdown():
        from app.db.session import async_engine
        await async_engine.dispose()
        logger.info("Application shut down")

    @application.get("/", tags=["health"])
    async def root():
        return {
            "service": settings.PROJECT_NAME,
            "version": settings.VERSION,
            "status": "healthy",
            "docs": "/docs",
        }

    @application.get("/health", tags=["health"])
    async def health():
        return {"status": "ok"}

    return application


app = create_app()
