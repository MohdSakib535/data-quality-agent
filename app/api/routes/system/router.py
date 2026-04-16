import json
import logging
import urllib.error
import urllib.request
from json import JSONDecodeError

import httpx
from fastapi import APIRouter, HTTPException

from app.core.config import settings

router = APIRouter(prefix="/system", tags=["system"])
logger = logging.getLogger(__name__)


def _describe_generation_error(exc: Exception) -> dict[str, object]:
    error_type = type(exc).__name__
    details: dict[str, object] = {
        "model_error_type": error_type,
        "model_error_status_code": None,
        "model_error_response": None,
    }

    if isinstance(exc, httpx.TimeoutException):
        details["model_error"] = (
            f"{error_type}: timed out after {settings.OLLAMA_TIMEOUT}s while calling Ollama /api/generate."
        )
        return details

    if isinstance(exc, httpx.HTTPStatusError):
        status_code = exc.response.status_code if exc.response is not None else None
        response_text = ""
        if exc.response is not None:
            try:
                response_text = exc.response.text.strip()
            except Exception:  # noqa: BLE001
                response_text = ""

        response_excerpt = response_text[:500] if response_text else None
        details["model_error_status_code"] = status_code
        details["model_error_response"] = response_excerpt
        details["model_error"] = (
            f"{error_type}: Ollama /api/generate returned HTTP {status_code}."
            if status_code is not None
            else f"{error_type}: Ollama /api/generate returned an HTTP error."
        )
        return details

    if isinstance(exc, httpx.RequestError):
        request_url = str(exc.request.url) if exc.request is not None else None
        details["model_error"] = (
            f"{error_type}: request to Ollama /api/generate failed"
            f"{f' for {request_url}' if request_url else ''}."
        )
        return details

    if isinstance(exc, JSONDecodeError):
        details["model_error"] = (
            f"{error_type}: Ollama /api/generate returned a non-JSON response."
        )
        return details

    message = str(exc).strip() or repr(exc)
    details["model_error"] = f"{error_type}: {message}"
    return details


@router.get("/ollama")
async def test_ollama_connection(prompt: str = "Reply with a short health check message."):
    """
    Verify Ollama is reachable and optionally return a small LLM reply for the given prompt.
    Echoes the provided prompt so callers can confirm which topic they sent.
    """
    tags_url = f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/tags"

    try:
        with urllib.request.urlopen(tags_url, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "ok": False,
                "message": "Could not connect to Ollama.",
                "base_url": settings.OLLAMA_BASE_URL,
                "model": settings.OLLAMA_MODEL,
                "error": str(exc),
            },
        ) from exc

    models = payload.get("models", [])
    available_models = [model.get("name", "") for model in models]
    matching_models = [
        model_name
        for model_name in available_models
        if model_name == settings.OLLAMA_MODEL or model_name.startswith(f"{settings.OLLAMA_MODEL}:")
    ]
    model_found = bool(matching_models)
    generate_model = matching_models[0] if matching_models else settings.OLLAMA_MODEL

    if not model_found:
        return {
            "ok": False,
            "base_url": settings.OLLAMA_BASE_URL,
            "model": settings.OLLAMA_MODEL,
            "model_found": False,
            "available_models": available_models,
            "prompt": prompt,
            "message": "Ollama is reachable, but the configured model was not found.",
        }

    model_reply = None
    model_error = None
    model_error_type = None
    model_error_status_code = None
    model_error_response = None
    
    try:
        generate_url = f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/generate"
        payload = {
            "model": generate_model,
            "prompt": prompt,
            "stream": False,
        }
        async with httpx.AsyncClient(timeout=settings.OLLAMA_TIMEOUT) as client:
            resp = await client.post(generate_url, json=payload)
            resp.raise_for_status()
            body = resp.json()
            model_reply = body.get("response")
    except Exception as exc:  # noqa: BLE001
        logger.exception("Ollama generate failed during health check.")
        error_details = _describe_generation_error(exc)
        model_error = str(error_details.get("model_error"))
        model_error_type = error_details.get("model_error_type")
        model_error_status_code = error_details.get("model_error_status_code")
        model_error_response = error_details.get("model_error_response")

    return {
        "ok": True,
        "base_url": settings.OLLAMA_BASE_URL,
        "model": settings.OLLAMA_MODEL,
        "model_used": generate_model,
        "model_found": True,
        "available_models": available_models,
        "prompt": prompt,
        "generation_ok": model_reply is not None,
        "model_reply": model_reply,
        "model_error": model_error,
        "model_error_type": model_error_type,
        "model_error_status_code": model_error_status_code,
        "model_error_response": model_error_response,
        "message": (
            "Ollama is reachable and generation succeeded."
            if model_reply is not None
            else "Ollama is reachable, but generation failed."
        ),
    }
