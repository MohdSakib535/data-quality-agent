"""
FastAPI middleware: structured request logging, error handling, timing.
"""
import logging
import time
import traceback
import uuid
from typing import Callable

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every request with timing, status, and a unique request ID."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = str(uuid.uuid4())[:8]
        start_time = time.time()

        # Attach request_id to state for downstream access
        request.state.request_id = request_id

        try:
            response = await call_next(request)
            duration_ms = round((time.time() - start_time) * 1000, 2)

            logger.info(
                "Request completed",
                extra={
                    "event": "http_request",
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration_ms": duration_ms,
                },
            )
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Duration-Ms"] = str(duration_ms)
            return response

        except Exception as exc:
            duration_ms = round((time.time() - start_time) * 1000, 2)
            logger.error(
                "Unhandled exception",
                extra={
                    "event": "http_error",
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "duration_ms": duration_ms,
                },
            )
            return JSONResponse(
                status_code=500,
                content={
                    "error": "internal_server_error",
                    "message": "An unexpected error occurred",
                    "request_id": request_id,
                },
            )


def setup_middleware(app: FastAPI) -> None:
    """Register all middleware on the FastAPI app."""
    app.add_middleware(RequestLoggingMiddleware)
