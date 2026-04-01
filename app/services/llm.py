"""
LLM service with 3-tier fallback strategy.

call_llm_with_fallback() is the ONLY way to call an LLM in this system.
Tier 1: Full prompt → validate JSON response
Tier 2: Simplified prompt, 2 retries with exponential backoff
Tier 3: Deterministic fallback (no LLM) — job still completes as "degraded"
"""
import json
import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
import redis

from app.core.config import settings

logger = logging.getLogger(__name__)

# Redis client for caching LLM responses
_redis: redis.Redis | None = None


def _get_redis() -> redis.Redis:
    global _redis
    if _redis is None:
        _redis = redis.from_url(settings.REDIS_URL, decode_responses=True)
    return _redis


@dataclass
class LLMResult:
    """Result from an LLM call, including fallback metadata."""
    data: dict | list
    tier_used: int = 1  # 1, 2, or 3
    llm_assisted: bool = True
    fallback_triggered: bool = False
    warnings: list[str] = field(default_factory=list)
    duration_ms: float = 0.0


def _cache_key(prompt: str, schema_fingerprint: str = "") -> str:
    """Generate a deterministic cache key."""
    raw = f"{prompt}|{schema_fingerprint}"
    return f"llm_cache:{hashlib.sha256(raw.encode()).hexdigest()}"


def _validate_json_response(text: str, expected_keys: list[str] | None = None) -> dict | None:
    """Parse and optionally validate that response JSON contains expected keys."""
    # Strip markdown code fences if present
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Remove first and last lines (fences)
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, dict):
        return None

    if expected_keys:
        missing = [k for k in expected_keys if k not in parsed]
        if missing:
            logger.warning("LLM response missing keys", extra={"missing": missing})
            return None

    return parsed


def _call_ollama(prompt: str, timeout: int) -> str:
    """Call Ollama API synchronously via httpx."""
    with httpx.Client(timeout=timeout) as client:
        response = client.post(
            f"{settings.OLLAMA_BASE_URL}/api/chat",
            json={
                "model": settings.OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": "You are a data engineering assistant. Return ONLY valid JSON. No markdown, no prose, no code fences."},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": 0.1,
                },
            },
        )
        response.raise_for_status()
        return response.json()["message"]["content"]


def _call_openai(prompt: str, timeout: int) -> str:
    """Call OpenAI API synchronously via httpx."""
    with httpx.Client(timeout=timeout) as client:
        response = client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": settings.OPENAI_MODEL,
                "messages": [
                    {"role": "system", "content": "You are a data engineering assistant. Return ONLY valid JSON. No markdown, no prose."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.1,
                "response_format": {"type": "json_object"},
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


def _call_anthropic(prompt: str, timeout: int) -> str:
    """Call Anthropic API synchronously via httpx."""
    with httpx.Client(timeout=timeout) as client:
        response = client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": settings.ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json={
                "model": settings.ANTHROPIC_MODEL,
                "max_tokens": 4096,
                "system": "You are a data engineering assistant. Return ONLY valid JSON. No markdown, no prose.",
                "messages": [
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.1,
            },
        )
        response.raise_for_status()
        return response.json()["content"][0]["text"]


def _call_llm_raw(prompt: str, timeout: int) -> str:
    """Route to the configured LLM provider."""
    if settings.LLM_PROVIDER == "ollama":
        return _call_ollama(prompt, timeout)
    elif settings.LLM_PROVIDER == "anthropic":
        return _call_anthropic(prompt, timeout)
    return _call_openai(prompt, timeout)


def _simplify_prompt(prompt: str) -> str:
    """
    Tier 2 simplification: strip examples, reduce verbosity.
    Keeps the core instruction and schema but removes sample data beyond 10 rows.
    """
    lines = prompt.split("\n")
    simplified_lines: list[str] = []
    in_sample_block = False
    sample_row_count = 0

    for line in lines:
        lower = line.lower().strip()
        # Detect sample data blocks
        if "sample_rows" in lower or "sample rows" in lower or "example" in lower:
            in_sample_block = True
            sample_row_count = 0
            simplified_lines.append(line)
            continue

        if in_sample_block:
            sample_row_count += 1
            if sample_row_count <= 10:
                simplified_lines.append(line)
            elif sample_row_count == 11:
                simplified_lines.append("  ... (truncated for brevity)")
            # Detect end of sample block
            if line.strip() in ("]", "],", "}", "},") and sample_row_count > 10:
                in_sample_block = False
                simplified_lines.append(line)
            continue

        simplified_lines.append(line)

    return "\n".join(simplified_lines)


def call_llm_with_fallback(
    prompt: str,
    expected_keys: list[str] | None = None,
    context: str = "general",
    fallback_response: dict | None = None,
    schema_fingerprint: str = "",
    use_cache: bool = True,
) -> LLMResult:
    """
    3-tier LLM call with fallback. This is the ONLY function that should call an LLM.

    Args:
        prompt: The full prompt to send.
        expected_keys: JSON keys that must be present in a valid response.
        context: Label for logging (e.g. "analysis", "cleaning", "nl_to_sql").
        fallback_response: Deterministic fallback dict if all LLM tiers fail.
        schema_fingerprint: For cache key generation.
        use_cache: Whether to check/store in Redis cache.

    Returns:
        LLMResult with data, tier_used, and metadata.
    """
    start_time = time.time()
    error_chain: list[str] = []

    # ── Check cache ──
    if use_cache:
        try:
            r = _get_redis()
            cache_k = _cache_key(prompt, schema_fingerprint)
            cached = r.get(cache_k)
            if cached:
                parsed = json.loads(cached)
                logger.info("LLM cache hit", extra={"context": context, "cache_key": cache_k[:16]})
                return LLMResult(
                    data=parsed,
                    tier_used=0,
                    llm_assisted=True,
                    duration_ms=(time.time() - start_time) * 1000,
                )
        except Exception as exc:
            logger.warning("Redis cache read failed", extra={"error": str(exc)})

    # ── Tier 1: Full prompt ──────────────────────────────────
    logger.info("LLM Tier 1 attempt", extra={"context": context})
    try:
        raw_response = _call_llm_raw(prompt, timeout=settings.LLM_TIMEOUT)
        parsed = _validate_json_response(raw_response, expected_keys)
        if parsed is not None:
            duration = (time.time() - start_time) * 1000
            logger.info(
                "LLM Tier 1 success",
                extra={"context": context, "duration_ms": duration},
            )
            # Cache the result
            if use_cache:
                try:
                    r = _get_redis()
                    r.setex(
                        _cache_key(prompt, schema_fingerprint),
                        settings.LLM_CACHE_TTL,
                        json.dumps(parsed),
                    )
                except Exception:
                    pass  # cache write failure is non-critical

            return LLMResult(data=parsed, tier_used=1, duration_ms=duration)
        else:
            error_chain.append("Tier 1: Invalid JSON response")
            logger.warning("LLM Tier 1 invalid JSON", extra={"context": context})
    except httpx.TimeoutException:
        error_chain.append("Tier 1: Timeout")
        logger.warning("LLM Tier 1 timeout", extra={"context": context})
    except httpx.HTTPStatusError as exc:
        error_chain.append(f"Tier 1: HTTP {exc.response.status_code}")
        logger.warning("LLM Tier 1 HTTP error", extra={"context": context, "status": exc.response.status_code})
    except Exception as exc:
        error_chain.append(f"Tier 1: {type(exc).__name__}: {str(exc)[:200]}")
        logger.warning("LLM Tier 1 error", extra={"context": context, "error": str(exc)[:200]})

    # ── Tier 2: Simplified prompt, 2 retries ─────────────────
    simplified = _simplify_prompt(prompt)
    backoff_delays = [2, 4]

    for attempt, delay in enumerate(backoff_delays, start=1):
        logger.info("LLM Tier 2 attempt", extra={"context": context, "attempt": attempt, "delay": delay})
        time.sleep(delay)
        try:
            raw_response = _call_llm_raw(simplified, timeout=settings.LLM_TIMEOUT)
            parsed = _validate_json_response(raw_response, expected_keys)
            if parsed is not None:
                duration = (time.time() - start_time) * 1000
                logger.info(
                    "LLM Tier 2 success",
                    extra={"context": context, "attempt": attempt, "duration_ms": duration},
                )
                if use_cache:
                    try:
                        r = _get_redis()
                        r.setex(
                            _cache_key(prompt, schema_fingerprint),
                            settings.LLM_CACHE_TTL,
                            json.dumps(parsed),
                        )
                    except Exception:
                        pass

                return LLMResult(
                    data=parsed,
                    tier_used=2,
                    llm_assisted=True,
                    fallback_triggered=True,
                    warnings=[f"Tier 2 fallback used after {len(error_chain)} Tier 1 failures"],
                    duration_ms=duration,
                )
            else:
                error_chain.append(f"Tier 2 attempt {attempt}: Invalid JSON")
        except httpx.TimeoutException:
            error_chain.append(f"Tier 2 attempt {attempt}: Timeout")
        except httpx.HTTPStatusError as exc:
            error_chain.append(f"Tier 2 attempt {attempt}: HTTP {exc.response.status_code}")
        except Exception as exc:
            error_chain.append(f"Tier 2 attempt {attempt}: {type(exc).__name__}: {str(exc)[:200]}")

    # ── Tier 3: Deterministic fallback ───────────────────────
    duration = (time.time() - start_time) * 1000
    logger.error(
        "LLM all tiers failed — using deterministic fallback",
        extra={"context": context, "error_chain": error_chain, "duration_ms": duration},
    )

    if fallback_response is None:
        fallback_response = {"error": "llm_unavailable", "fallback": True}

    return LLMResult(
        data=fallback_response,
        tier_used=3,
        llm_assisted=False,
        fallback_triggered=True,
        warnings=[
            "All LLM tiers failed — deterministic fallback applied",
            f"Error chain: {'; '.join(error_chain)}",
        ],
        duration_ms=duration,
    )
