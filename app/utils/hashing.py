"""
SHA-256 hashing helpers for cache keys and schema fingerprints.
"""
import hashlib
import json
from typing import Any


def sha256_hex(data: str) -> str:
    """Return SHA-256 hex digest of a string."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def schema_fingerprint(schema: dict[str, str]) -> str:
    """
    Generate a deterministic fingerprint from a schema dict.
    Sorts column names + dtypes to ensure consistency regardless of column order.
    """
    sorted_items = sorted(schema.items())
    raw = json.dumps(sorted_items, sort_keys=True)
    return sha256_hex(raw)


def prompt_cache_key(dataset_id: str, prompt: str) -> str:
    """
    Generate a Redis cache key from dataset_id + prompt.
    Used to cache cleaning results so re-running the same prompt is instant.
    """
    raw = f"{dataset_id}|{prompt}"
    return f"clean_cache:{sha256_hex(raw)}"


def content_hash(data: bytes) -> str:
    """SHA-256 of raw byte content."""
    return hashlib.sha256(data).hexdigest()
