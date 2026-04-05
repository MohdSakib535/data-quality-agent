import logging

import httpx
from fastapi import HTTPException

from app.services.object_storage import ObjectStorageError


def raise_http_for_service_error(
    exc: Exception,
    *,
    operation: str,
    logger: logging.Logger,
) -> None:
    if isinstance(exc, FileNotFoundError):
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if isinstance(exc, ValueError):
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if isinstance(exc, (TimeoutError, httpx.TimeoutException)):
        logger.exception("%s timed out: %s", operation, exc)
        raise HTTPException(
            status_code=504,
            detail=f"{operation} timed out. Please retry.",
        ) from exc

    if isinstance(exc, (ConnectionError, httpx.HTTPError, ObjectStorageError)):
        logger.exception("%s dependency failure: %s", operation, exc)
        raise HTTPException(
            status_code=503,
            detail=f"{operation} is temporarily unavailable. Please retry.",
        ) from exc

    logger.exception("%s failed: %s", operation, exc)
    raise HTTPException(
        status_code=500,
        detail=f"{operation} failed due to an internal error. Please retry.",
    ) from exc
