import json
import logging
import urllib.error
import urllib.request

import httpx
from fastapi import APIRouter, HTTPException

from app.core.config import settings

router = APIRouter(prefix="/system", tags=["system"])
logger = logging.getLogger(__name__)


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
        logger.warning("Ollama generate failed during health check: %s", exc)
        model_error = str(exc)

    return {
        "ok": True,
        "base_url": settings.OLLAMA_BASE_URL,
        "model": settings.OLLAMA_MODEL,
        "model_used": generate_model,
        "model_found": True,
        "available_models": available_models,
        "prompt": prompt,
        "model_reply": model_reply,
        "model_error": model_error,
        "message": (
            "Ollama is reachable and generation succeeded."
            if model_reply is not None
            else "Ollama is reachable (generation skipped or failed)."
        ),
    }
