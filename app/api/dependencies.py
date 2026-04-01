"""
FastAPI dependencies: DB session, S3 client injection.
"""
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db


async def get_database(db: AsyncSession = Depends(get_db)) -> AsyncSession:
    """Inject async DB session into route handlers."""
    return db
