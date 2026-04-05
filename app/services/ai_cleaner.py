import json
import re
import ast
import logging
import urllib.error
import urllib.request
from typing import List, Dict, Any
from functools import lru_cache
import time

import httpx
import pandas as pd
from langchain_core.output_parsers import StrOutputParser
from app.prompts.cleaning_prompts import clean_data_prompt_template
from app.schemas.job import DataSuggestion, DatasetAnalysisResponse
from app.core.config import settings
from app.services.deterministic_cleaner import (
    NUMERIC_COLUMN_HINTS,
    PHONE_COLUMN_HINTS,
    PHONE_PATTERN,
    _to_snake_case,
    compute_quality_score,
    generate_rule_based_suggestions,
)
from app.services.llm_privacy import redact_analysis_profile, select_llm_safe_columns

try:
    from langchain_ollama import ChatOllama
except ImportError:
    from langchain_community.chat_models import ChatOllama

logger = logging.getLogger(__name__)

ANALYSIS_LLM_TIMEOUT_SECONDS = 20
ANALYSIS_LLM_OPTIONS = {
    "temperature": 0,
    "num_predict": 400,
}
ANALYSIS_SYSTEM_PROMPT = (
    "You are a strict data quality issue summarizer. "
    "You will receive a deterministic dataset profile and a few sample rows. "
    "Return JSON only with this exact schema: "
    '{"suggestions":[{"issue_description":"...","priority":"High|Medium|Low","resolution_prompt":"..."}]}. '
    "Do not output markdown. Do not output prose. Do not compute quality_score. "
    "Use only the provided detected issues and profile metrics. "
    "Keep suggestions dataset-level and generic. Return at most 5 suggestions."
)


def _coerce_python_payload(candidate: str) -> Any:
    """Best-effort parsing for python-like payloads (single quotes, True/False/None)."""
    payload = ast.literal_eval(candidate)
    if isinstance(payload, (dict, list)):
        return payload
    raise ValueError(f"Unsupported payload type: {type(payload)}")


def _extract_json_payload(raw_response: str) -> Any:
    """Recover JSON payloads when the model wraps them in prose or markdown fences."""
    candidates = []
    fenced_match = re.search(r"```json\s*(.*?)```", raw_response, re.DOTALL | re.IGNORECASE)
    if fenced_match:
        candidates.append(fenced_match.group(1).strip())

    fenced_match_2 = re.search(r"```\s*(.*?)```", raw_response, re.DOTALL | re.IGNORECASE)
    if fenced_match_2:
        candidates.append(fenced_match_2.group(1).strip())

    stripped = raw_response.strip()
    candidates.append(stripped)

    array_match = re.search(r"(\[\s*.*\s*\])", raw_response, re.DOTALL)
    if array_match:
        candidates.append(array_match.group(1).strip())

    object_match = re.search(r"(\{\s*.*\s*\})", raw_response, re.DOTALL)
    if object_match:
        candidates.append(object_match.group(1).strip())

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
            return payload
        except json.JSONDecodeError:
            pass

        try:
            return _coerce_python_payload(candidate)
        except Exception:
            continue

    raise ValueError("Model response did not contain a valid JSON payload.")

def _normalize_analysis_suggestion(suggestion: DataSuggestion) -> DataSuggestion:
    return DataSuggestion(
        issue_description=suggestion.issue_description.strip(),
        priority=suggestion.priority.strip(),
        resolution_prompt=suggestion.resolution_prompt.strip(),
    )


@lru_cache(maxsize=16)
def _resolve_ollama_model_name_cached(
    base_url: str,
    configured_model: str,
    timeout: int,
) -> str:
    """
    Resolve the configured Ollama model to an installed model name, allowing
    tagged variants such as `model:latest` or `model:7b`.
    """
    tags_url = f"{base_url.rstrip('/')}/api/tags"

    try:
        with urllib.request.urlopen(tags_url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return configured_model

    available_models = [model.get("name", "") for model in payload.get("models", [])]
    for model_name in available_models:
        if model_name == configured_model or model_name.startswith(f"{configured_model}:"):
            return model_name

    return configured_model


def _resolve_ollama_model_name() -> str:
    return _resolve_ollama_model_name_cached(
        settings.OLLAMA_BASE_URL,
        settings.OLLAMA_MODEL,
        settings.OLLAMA_TIMEOUT,
    )


def warm_analysis_runtime_cache() -> None:
    _resolve_ollama_model_name()


def _build_analysis_llm_payload(profile: dict[str, Any]) -> dict[str, Any]:
    redacted_profile = redact_analysis_profile(profile)
    payload = {
        "row_count_sampled": redacted_profile.get("row_count_sampled", 0),
        "quality_score": redacted_profile.get("quality_score", 0),
        "duplicate_row_percent": redacted_profile.get("duplicate_row_percent", 0.0),
        "columns": redacted_profile.get("columns", []),
        "dataset_issues": redacted_profile.get("dataset_issues", []),
        "sample_rows": redacted_profile.get("sample_rows", [])[:5],
    }
    return payload


def _build_analysis_prompt(profile: dict[str, Any]) -> str:
    llm_payload = _build_analysis_llm_payload(profile)
    return (
        f"{ANALYSIS_SYSTEM_PROMPT}\n\n"
        f"Profile JSON:\n{json.dumps(llm_payload, ensure_ascii=False, separators=(',', ':'))}\n"
    )


async def _request_analysis_suggestions_from_llm(profile: dict[str, Any]) -> list[DataSuggestion]:
    resolved_model = _resolve_ollama_model_name()
    payload = {
        "model": resolved_model,
        "prompt": _build_analysis_prompt(profile),
        "stream": False,
        "format": "json",
        "options": ANALYSIS_LLM_OPTIONS,
    }
    async with httpx.AsyncClient(timeout=ANALYSIS_LLM_TIMEOUT_SECONDS) as client:
        response = await client.post(
            f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/generate",
            json=payload,
        )
        response.raise_for_status()
        body = response.json()

    result_dict = _extract_json_payload(body.get("response", ""))
    suggestions_payload = result_dict.get("suggestions", result_dict) if isinstance(result_dict, dict) else result_dict
    if not isinstance(suggestions_payload, list):
        raise ValueError("Model response did not contain a suggestions list.")

    suggestions = [
        _normalize_analysis_suggestion(DataSuggestion(**suggestion))
        for suggestion in suggestions_payload[:5]
    ]
    return suggestions


def _normalize_cleaned_batch_payload(cleaned_batch: Any) -> Any:
    if isinstance(cleaned_batch, dict):
        for value in cleaned_batch.values():
            if isinstance(value, list):
                return value
    return cleaned_batch


def _prompt_mentions_missing_values(user_prompt: str) -> bool:
    normalized_prompt = user_prompt.strip().lower()
    missing_terms = [
        "missing",
        "null",
        "blank",
        "empty",
        "n/a",
        "placeholder",
    ]
    return any(term in normalized_prompt for term in missing_terms)


def _apply_text_cleaning(
    df: pd.DataFrame,
    *,
    normalize_missing: bool = False,
    trim_whitespace: bool = False,
) -> pd.DataFrame:
    if not normalize_missing and not trim_whitespace:
        return df

    null_tokens = {
        token.strip().lower()
        for token in settings.NULL_TOKENS.split(",")
        if token.strip()
    } if normalize_missing else set()

    normalized_df = df.copy()
    for column in normalized_df.columns:
        series = normalized_df[column]
        if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
            continue

        def _transform_value(value):
            if not isinstance(value, str):
                return value

            transformed = value.strip() if trim_whitespace else value
            if normalize_missing and transformed.strip().lower() in null_tokens:
                return pd.NA
            return transformed

        normalized_df[column] = series.map(_transform_value)

    return normalized_df


def _trim_whitespace_in_string_cells(df: pd.DataFrame) -> pd.DataFrame:
    return _apply_text_cleaning(df, trim_whitespace=True)


def _normalize_missing_tokens(df: pd.DataFrame) -> pd.DataFrame:
    return _apply_text_cleaning(df, normalize_missing=True)


def _should_remove_exact_duplicates(user_prompt: str) -> bool:
    normalized_prompt = user_prompt.strip().lower()

    duplicate_terms = [
        "duplicate",
        "duplicates",
        "deduplicate",
        "dedup",
        "redundant copies",
    ]
    removal_terms = [
        "remove",
        "drop",
        "delete",
        "keep one",
        "canonical",
    ]

    return (
        any(term in normalized_prompt for term in duplicate_terms)
        and any(term in normalized_prompt for term in removal_terms)
    ) or "exact duplicate" in normalized_prompt


def _remove_exact_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().reset_index(drop=True)


def _remove_exact_duplicate_rows_across_chunks(
    df: pd.DataFrame,
    seen_row_hashes: set[int],
) -> pd.DataFrame:
    if df.empty:
        return df

    comparable_df = df.astype(object).where(pd.notnull(df), settings.NULL_OUTPUT_TOKEN)
    row_hashes = pd.util.hash_pandas_object(comparable_df.astype(str), index=False)

    keep_mask = []
    local_hashes: set[int] = set()

    for row_hash in row_hashes.tolist():
        if row_hash in seen_row_hashes or row_hash in local_hashes:
            keep_mask.append(False)
            continue

        keep_mask.append(True)
        local_hashes.add(row_hash)

    seen_row_hashes.update(local_hashes)
    return df.loc[keep_mask].reset_index(drop=True)


def _should_normalize_headers(user_prompt: str) -> bool:
    normalized_prompt = user_prompt.strip().lower()
    header_terms = [
        "header",
        "headers",
        "column name",
        "column names",
        "snake_case",
    ]
    return any(term in normalized_prompt for term in header_terms)


def _should_trim_whitespace(user_prompt: str) -> bool:
    normalized_prompt = user_prompt.strip().lower()
    whitespace_terms = [
        "whitespace",
        "leading or trailing",
        "trim",
        "strip spaces",
    ]
    return any(term in normalized_prompt for term in whitespace_terms)


def _is_missing_value_only_prompt(user_prompt: str) -> bool:
    normalized_prompt = user_prompt.strip().lower()
    missing_terms = [
        "missing",
        "null",
        "blank",
        "empty",
        "n/a",
        "placeholder",
    ]
    ai_terms = [
        "standardize spelling",
        "correct spelling",
        "category",
        "map",
        "capitalize",
        "capitalization",
        "typo",
        "canonical form",
    ]
    return any(term in normalized_prompt for term in missing_terms) and not any(
        term in normalized_prompt for term in ai_terms
    )


def _is_date_only_prompt(user_prompt: str) -> bool:
    """
    Identify prompts that only ask for date/datetime normalization, which can be
    handled deterministically without LLM calls.
    """
    normalized_prompt = user_prompt.strip().lower()
    date_terms = [
        "date format",
        "dates",
        "date column",
        "date columns",
        "datetime",
        "timestamp",
        "date/time",
    ]
    ai_terms = [
        "standardize spelling",
        "correct spelling",
        "category",
        "map",
        "typo",
        "capitalize",
        "capitalization",
        "text",
        "string",
    ]
    return any(term in normalized_prompt for term in date_terms) and not any(
        term in normalized_prompt for term in ai_terms
    )


def _is_phone_only_prompt(user_prompt: str) -> bool:
    normalized_prompt = user_prompt.strip().lower()
    phone_terms = [
        "phone",
        "phones",
        "mobile",
        "contact number",
        "country code",
        "phone-like",
        "punctuation noise",
    ]
    ai_terms = [
        "standardize spelling",
        "correct spelling",
        "category",
        "map",
        "typo",
        "capitalize",
        "capitalization",
        "name",
        "address",
        "email",
        "date",
    ]
    return any(term in normalized_prompt for term in phone_terms) and not any(
        term in normalized_prompt for term in ai_terms
    )


def _should_keep_only_valid_phone_rows(user_prompt: str) -> bool:
    normalized_prompt = user_prompt.strip().lower()
    keep_terms = [
        "keep only valid",
        "only valid",
        "remove invalid",
        "drop invalid",
        "valid phone-like",
        "valid phone",
    ]
    return any(term in normalized_prompt for term in keep_terms)


def _is_text_normalization_only_prompt(user_prompt: str) -> bool:
    normalized_prompt = user_prompt.strip().lower()
    text_terms = [
        "trim leading and trailing whitespace",
        "leading and trailing whitespace",
        "standardize casing",
        "standardize case",
        "whitespace inconsistencies",
        "casing inconsistencies",
        "repeated text values",
        "text normalization",
        "normalize text",
        "whitespace",
    ]
    ai_terms = [
        "category mapping",
        "translate",
        "summarize",
        "rewrite",
        "name formatting",
        "address formatting",
    ]
    return any(term in normalized_prompt for term in text_terms) and not any(
        term in normalized_prompt for term in ai_terms
    )


def _is_numeric_only_prompt(user_prompt: str) -> bool:
    normalized_prompt = user_prompt.strip().lower()
    numeric_terms = [
        "numeric-like",
        "numeric",
        "numbers",
        "currency symbols",
        "separators",
        "parse cleanly as numbers",
        "formatting noise",
    ]
    ai_terms = [
        "category",
        "map",
        "name",
        "address",
        "text",
        "free text",
    ]
    return any(term in normalized_prompt for term in numeric_terms) and not any(
        term in normalized_prompt for term in ai_terms
    )


def _is_duplicate_only_prompt(user_prompt: str) -> bool:
    normalized_prompt = user_prompt.strip().lower()
    duplicate_terms = [
        "duplicate",
        "duplicates",
        "deduplicate",
        "dedup",
        "redundant copies",
    ]
    ai_terms = [
        "standardize spelling",
        "correct spelling",
        "category",
        "map",
        "capitalize",
        "capitalization",
        "typo",
    ]
    return any(term in normalized_prompt for term in duplicate_terms) and not any(
        term in normalized_prompt for term in ai_terms
    )


def _requires_ai_cleaning(user_prompt: str) -> bool:
    normalized_prompt = user_prompt.strip().lower()
    deterministic_only = (
        _is_duplicate_only_prompt(user_prompt)
        or _is_missing_value_only_prompt(user_prompt)
        or _is_date_only_prompt(user_prompt)
        or _is_phone_only_prompt(user_prompt)
        or _is_text_normalization_only_prompt(user_prompt)
        or _is_numeric_only_prompt(user_prompt)
        or _should_normalize_headers(user_prompt)
        or _should_trim_whitespace(user_prompt)
    )
    if deterministic_only:
        return False

    ai_terms = [
        "standardize",
        "correct",
        "fix typo",
        "normalize values",
        "map",
        "category",
        "categorize",
        "capitalization",
        "capitalize",
        "spelling",
        "format names",
    ]

    if any(term in normalized_prompt for term in ai_terms):
        return True

    return True


def _normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    normalized_df = df.copy()
    normalized_df.columns = [_to_snake_case(str(column)) for column in normalized_df.columns]
    return normalized_df


def _normalize_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Try to normalize date-like string columns to the configured output format
    without relying on the LLM. We only transform columns where a majority of
    non-null values parse as dates to avoid false positives.
    """
    normalized_df = df.copy()
    for column in normalized_df.columns:
        series = normalized_df[column]
        if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
            continue

        series_non_null = series[series.notna()]
        if series_non_null.empty:
            continue

        try:
            parsed = pd.to_datetime(
                series_non_null,
                errors="coerce",
                infer_datetime_format=True,
                utc=False,
            )
        except TypeError:
            # pandas >= 2.0 removed infer_datetime_format; fall back cleanly.
            parsed = pd.to_datetime(
                series_non_null,
                errors="coerce",
                utc=False,
            )
        success_ratio = parsed.notna().mean()
        # Require a reasonable share of valid parses to avoid changing non-date columns.
        if success_ratio < 0.5:
            continue

        formatted = parsed.dt.strftime(settings.DATE_OUTPUT_FORMAT)
        updated_series = series.copy()
        updated_series.loc[formatted.index] = formatted
        normalized_df[column] = updated_series

    return normalized_df


def _is_phone_candidate_column(column_name: str, series: pd.Series) -> bool:
    normalized_name = column_name.strip().lower()
    if any(hint in normalized_name for hint in PHONE_COLUMN_HINTS):
        return True

    string_series = series.astype("string")
    non_null_values = string_series[series.notna()].str.strip()
    if non_null_values.empty:
        return False

    phone_digits = non_null_values.str.replace(r"\D+", "", regex=True)
    validity_ratio = float(phone_digits.str.fullmatch(PHONE_PATTERN, na=False).mean())
    return validity_ratio >= 0.6


def _normalize_phone_value(value: Any) -> tuple[Any, bool]:
    if pd.isna(value):
        return value, True

    text = str(value).strip()
    if not text:
        return value, True

    digits = re.sub(r"\D+", "", text)
    if not digits or not PHONE_PATTERN.fullmatch(digits):
        return value, False

    has_explicit_country_code = text.startswith("+") or len(digits) > 10
    normalized = f"+{digits}" if has_explicit_country_code else digits
    return normalized, True


def _normalize_phone_columns(df: pd.DataFrame, *, keep_only_valid_rows: bool = False) -> pd.DataFrame:
    normalized_df = df.copy()
    keep_mask = pd.Series(True, index=normalized_df.index)

    for column in normalized_df.columns:
        series = normalized_df[column]
        if not (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or pd.api.types.is_numeric_dtype(series)
        ):
            continue

        if not _is_phone_candidate_column(str(column), series):
            continue

        normalized_values: list[Any] = []
        valid_flags: list[bool] = []

        for value in series.tolist():
            normalized_value, is_valid = _normalize_phone_value(value)
            normalized_values.append(normalized_value)
            valid_flags.append(is_valid)

        normalized_df[column] = normalized_values

        if keep_only_valid_rows:
            non_empty_mask = series.notna() & series.astype("string").str.strip().ne("")
            valid_series = pd.Series(valid_flags, index=normalized_df.index)
            keep_mask &= (~non_empty_mask) | valid_series

    if keep_only_valid_rows:
        normalized_df = normalized_df.loc[keep_mask].reset_index(drop=True)

    return normalized_df


def _is_moderate_cardinality_text_column(series: pd.Series) -> bool:
    string_series = series.astype("string")
    non_null_values = string_series[series.notna()].str.strip()
    if non_null_values.empty:
        return False

    unique_count = int(non_null_values.nunique(dropna=True))
    unique_ratio = unique_count / max(len(non_null_values), 1)
    average_length = float(non_null_values.str.len().fillna(0).mean())
    return unique_count <= 200 and unique_ratio <= 0.5 and average_length <= 80


def _canonical_text_variant(values: pd.Series) -> str:
    trimmed_values = values.astype("string").dropna().str.strip()
    trimmed_values = trimmed_values[trimmed_values.ne("")]
    if trimmed_values.empty:
        return ""

    counts = trimmed_values.value_counts()
    if not counts.empty:
        return str(counts.index[0])
    return str(trimmed_values.iloc[0])


def _normalize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized_df = _apply_text_cleaning(df, trim_whitespace=True)

    for column in normalized_df.columns:
        series = normalized_df[column]
        if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
            continue

        if not _is_moderate_cardinality_text_column(series):
            continue

        string_series = series.astype("string")
        trimmed_series = string_series.str.strip()
        normalized_keys = trimmed_series.str.lower()
        non_empty_mask = trimmed_series.notna() & trimmed_series.ne("")
        if not non_empty_mask.any():
            continue

        canonical_map: dict[str, str] = {}
        for key, group in trimmed_series[non_empty_mask].groupby(normalized_keys[non_empty_mask]):
            if key is pd.NA or key is None or str(key).strip() == "":
                continue
            canonical_map[str(key)] = _canonical_text_variant(group)

        normalized_df[column] = [
            canonical_map.get(str(key), value) if pd.notna(key) else value
            for value, key in zip(trimmed_series.tolist(), normalized_keys.tolist())
        ]

    return normalized_df


def _is_numeric_candidate_column(column_name: str, series: pd.Series) -> bool:
    normalized_name = column_name.strip().lower()
    if any(hint in normalized_name for hint in NUMERIC_COLUMN_HINTS):
        return True

    string_series = series.astype("string")
    non_null_values = string_series[series.notna()].str.strip()
    if non_null_values.empty:
        return False

    cleaned_values = non_null_values.str.replace(r"[,\s$₹€£%]", "", regex=True)
    parsed = pd.to_numeric(cleaned_values, errors="coerce")
    return float(parsed.notna().mean()) >= 0.6


def _format_numeric_value(value: float) -> str:
    if pd.isna(value):
        return ""
    if float(value).is_integer():
        return str(int(value))
    return format(float(value), "f").rstrip("0").rstrip(".")


def _normalize_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized_df = df.copy()

    for column in normalized_df.columns:
        series = normalized_df[column]
        if not (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or pd.api.types.is_numeric_dtype(series)
        ):
            continue

        if not _is_numeric_candidate_column(str(column), series):
            continue

        string_series = series.astype("string")
        non_empty_mask = series.notna() & string_series.str.strip().ne("")
        if not non_empty_mask.any():
            continue

        cleaned_values = string_series.str.replace(r"[,\s$₹€£%]", "", regex=True)
        parsed = pd.to_numeric(cleaned_values.where(non_empty_mask), errors="coerce")
        formatted = parsed.map(_format_numeric_value)

        updated_series = string_series.copy()
        valid_mask = non_empty_mask & parsed.notna()
        updated_series.loc[valid_mask] = formatted.loc[valid_mask]
        normalized_df[column] = updated_series

    return normalized_df


def _build_cleaning_chain():
    llm_kwargs = {
        "model": settings.OLLAMA_MODEL,
        "base_url": settings.OLLAMA_BASE_URL,
        "temperature": 0.0,
    }
    try:
        llm = ChatOllama(format="json", **llm_kwargs)
    except TypeError:
        llm = ChatOllama(**llm_kwargs)
    return clean_data_prompt_template | llm | StrOutputParser()


def build_cleaning_chain():
    return _build_cleaning_chain()


def prompt_requires_ai_cleaning(user_prompt: str) -> bool:
    return _requires_ai_cleaning(user_prompt)


def prompt_removes_exact_duplicates(user_prompt: str) -> bool:
    return _should_remove_exact_duplicates(user_prompt)


def _normalize_text_for_match(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _extract_target_columns_from_prompt(columns: list[str], user_prompt: str) -> set[str]:
    normalized_prompt = f" {_normalize_text_for_match(user_prompt)} "
    target_columns: set[str] = set()

    for column in columns:
        column_label = str(column).strip()
        if not column_label:
            continue

        variants = {
            column_label.lower(),
            column_label.lower().replace("_", " "),
            _to_snake_case(column_label).replace("_", " "),
            _normalize_text_for_match(column_label),
        }
        if any(variant and f" {variant} " in normalized_prompt for variant in variants):
            target_columns.add(column_label)

    return target_columns


def _merge_cleaned_row(
    original_row: dict[str, Any],
    cleaned_row: Any,
    target_columns: set[str],
) -> dict[str, Any]:
    if not isinstance(cleaned_row, dict):
        return dict(original_row)

    merged = dict(original_row)
    columns_to_apply = target_columns or set(original_row.keys())
    for column in columns_to_apply:
        if column in cleaned_row:
            merged[column] = cleaned_row[column]
    return merged


def _build_ai_view_row(
    original_row: dict[str, Any],
    ai_columns: list[str],
) -> dict[str, Any]:
    if not ai_columns:
        return dict(original_row)
    return {column: original_row.get(column) for column in ai_columns}


def _compute_record_hashes(
    records: list[dict[str, Any]],
    *,
    columns: list[str],
) -> list[int]:
    if not records:
        return []

    comparable_df = pd.DataFrame(records, columns=columns)
    comparable_df = comparable_df.astype(object).where(pd.notnull(comparable_df), settings.NULL_OUTPUT_TOKEN)
    return (
        pd.util.hash_pandas_object(comparable_df.astype(str), index=False)
        .astype("uint64")
        .tolist()
    )


def _invoke_cleaning_batch(
    chain,
    batch_json: str,
    user_prompt: str,
    expected_rows: int,
) -> List[Dict[str, Any]]:
    retry_prompts = [
        user_prompt,
        (
            f"{user_prompt}\n\n"
            f"Important: return only a JSON array with exactly {expected_rows} objects, keep all original keys, "
            "and if the instruction asks for a literal placeholder like NaN then output the string value \"NaN\"."
        ),
    ]
    last_error: Exception | None = None

    for prompt in retry_prompts:
        try:
            raw_response = chain.invoke({
                "batch_json": batch_json,
                "user_prompt": prompt,
            })

            cleaned_batch = _normalize_cleaned_batch_payload(_extract_json_payload(raw_response))
            if not isinstance(cleaned_batch, list):
                raise ValueError(f"Expected a list of JSON objects, got {type(cleaned_batch)}")
            if len(cleaned_batch) != expected_rows:
                raise ValueError(f"Batch size mismatch. Expected {expected_rows}, got {len(cleaned_batch)}")
            return cleaned_batch
        except Exception as exc:
            last_error = exc

    raise last_error if last_error is not None else ValueError("Cleaning batch failed")


def _invoke_cleaning_batch_with_fallback(
    chain,
    batch_records: list[dict[str, Any]],
    user_prompt: str,
) -> list[dict[str, Any]]:
    expected_rows = len(batch_records)
    if expected_rows == 0:
        return []

    batch_json = json.dumps(batch_records, ensure_ascii=False, separators=(",", ":"))

    try:
        return _invoke_cleaning_batch(
            chain=chain,
            batch_json=batch_json,
            user_prompt=user_prompt,
            expected_rows=expected_rows,
        )
    except Exception:
        if expected_rows == 1:
            # Do not fail the entire cleaning run for a single bad model output.
            # Keep the original row unchanged in this case.
            return [dict(batch_records[0])]

        mid = expected_rows // 2
        left_records = batch_records[:mid]
        right_records = batch_records[mid:]

        try:
            left_cleaned = _invoke_cleaning_batch_with_fallback(
                chain,
                left_records,
                user_prompt,
            )
        except Exception:
            left_cleaned = [dict(row) for row in left_records]

        try:
            right_cleaned = _invoke_cleaning_batch_with_fallback(
                chain,
                right_records,
                user_prompt,
            )
        except Exception:
            right_cleaned = [dict(row) for row in right_records]

        return left_cleaned + right_cleaned


def _clean_column_values_with_ai(
    df: pd.DataFrame,
    *,
    ai_columns: list[str],
    user_prompt: str,
    chain,
    ai_value_cache: dict[str, Any] | None = None,
) -> pd.DataFrame:
    cleaned_df = df.copy()
    batch_size = max(1, settings.AI_VALUE_BATCH_SIZE)

    for column in ai_columns:
        if column not in cleaned_df.columns:
            continue

        series = cleaned_df[column]
        string_series = series.astype("string")
        non_empty_values = string_series[series.notna()].str.strip()
        non_empty_values = non_empty_values[non_empty_values.ne("")]
        unique_values = non_empty_values.drop_duplicates().tolist()

        if not unique_values:
            continue

        unresolved_values = [
            value
            for value in unique_values
            if ai_value_cache is None or f"{column}::{value}" not in ai_value_cache
        ]

        if len(unresolved_values) > settings.AI_VALUE_CLEAN_MAX_UNIQUE_VALUES:
            raise ValueError(
                f"AI cleaning for column '{column}' is too large for a scoped value-mapping pass. "
                "Narrow the prompt further or use a deterministic cleaning instruction."
            )

        value_map: dict[str, Any] = {}
        if ai_value_cache is not None:
            for value in unique_values:
                cache_key = f"{column}::{value}"
                if cache_key in ai_value_cache:
                    value_map[value] = ai_value_cache[cache_key]

        for start in range(0, len(unresolved_values), batch_size):
            batch_values = unresolved_values[start:start + batch_size]
            batch_records = [{column: value} for value in batch_values]
            cleaned_batch = _invoke_cleaning_batch_with_fallback(
                chain=chain,
                batch_records=batch_records,
                user_prompt=user_prompt,
            )

            for original_record, cleaned_record in zip(batch_records, cleaned_batch):
                original_value = original_record[column]
                mapped_value = (
                    cleaned_record.get(column, original_value)
                    if isinstance(cleaned_record, dict)
                    else original_value
                )
                value_map[str(original_value)] = mapped_value
                if ai_value_cache is not None:
                    ai_value_cache[f"{column}::{original_value}"] = mapped_value

        cleaned_df[column] = [
            value_map.get(str(value).strip(), value)
            if pd.notna(value) and str(value).strip()
            else value
            for value in series.tolist()
        ]

    return cleaned_df


async def analyze_dataset(profile: dict[str, Any]) -> tuple[DatasetAnalysisResponse, float]:
    """
    Summarize a deterministic profile into user-facing suggestions.
    The quality score is computed deterministically and never by the LLM.
    """
    quality_score = compute_quality_score(profile)
    enriched_profile = {**profile, "quality_score": quality_score}

    llm_start = time.perf_counter()
    try:
        suggestions = await _request_analysis_suggestions_from_llm(enriched_profile)
    except Exception as exc:
        logger.warning("Falling back to rule-based analysis suggestions: %s", exc)
        suggestions = generate_rule_based_suggestions(enriched_profile)
    llm_request_ms = (time.perf_counter() - llm_start) * 1000

    response = DatasetAnalysisResponse(
        job_id="",
        quality_score=quality_score,
        suggestions=suggestions,
    )
    return response, llm_request_ms

def clean_dataset_with_prompt(df: pd.DataFrame, user_prompt: str) -> pd.DataFrame:
    """
    Clean a single dataframe chunk using deterministic transforms and, when required,
    an LLM batch loop.
    """
    return clean_dataframe_chunk(df, user_prompt)


def clean_dataframe_chunk(
    df: pd.DataFrame,
    user_prompt: str,
    *,
    chain=None,
    seen_row_hashes: set[int] | None = None,
    ai_row_cache: dict[int, dict[str, Any]] | None = None,
) -> pd.DataFrame:
    should_normalize_missing = _prompt_mentions_missing_values(user_prompt)
    should_trim_whitespace = _should_trim_whitespace(user_prompt)
    cleaned_df = _apply_text_cleaning(
        df,
        normalize_missing=should_normalize_missing,
        trim_whitespace=should_trim_whitespace,
    )

    if _should_normalize_headers(user_prompt):
        cleaned_df = _normalize_headers(cleaned_df)

    if _is_date_only_prompt(user_prompt):
        cleaned_df = _normalize_date_columns(cleaned_df)

    if _is_phone_only_prompt(user_prompt):
        cleaned_df = _normalize_phone_columns(
            cleaned_df,
            keep_only_valid_rows=_should_keep_only_valid_phone_rows(user_prompt),
        )

    if _is_text_normalization_only_prompt(user_prompt):
        cleaned_df = _normalize_text_columns(cleaned_df)

    if _is_numeric_only_prompt(user_prompt):
        cleaned_df = _normalize_numeric_columns(cleaned_df)

    if _should_remove_exact_duplicates(user_prompt):
        if seen_row_hashes is None:
            cleaned_df = _remove_exact_duplicate_rows(cleaned_df)
        else:
            cleaned_df = _remove_exact_duplicate_rows_across_chunks(cleaned_df, seen_row_hashes)

    if cleaned_df.empty or not _requires_ai_cleaning(user_prompt):
        return cleaned_df

    active_chain = chain or _build_cleaning_chain()
    batch_size = min(
        settings.AI_BATCH_SIZE,
        max(10, 1200 // max(len(cleaned_df.columns), 1)),
    )
    target_columns = _extract_target_columns_from_prompt(
        [str(column) for column in cleaned_df.columns],
        user_prompt,
    )
    ai_columns = (
        [column for column in cleaned_df.columns if str(column) in target_columns]
        if target_columns
        else [str(column) for column in cleaned_df.columns]
    )
    requested_ai_columns = list(ai_columns)
    ai_columns = select_llm_safe_columns(cleaned_df, ai_columns)

    if not ai_columns:
        if requested_ai_columns:
            logger.info(
                "Skipping raw LLM cleaning because all candidate columns were privacy-blocked: %s",
                requested_ai_columns,
            )
        return cleaned_df

    if (
        len(cleaned_df) >= settings.AI_ROW_LEVEL_LARGE_DATASET_THRESHOLD
        and not target_columns
    ):
        raise ValueError(
            "For large datasets, AI cleaning prompts must mention the target column names explicitly "
            "or use a deterministic cleaning instruction."
        )

    if target_columns and len(ai_columns) <= settings.AI_VALUE_CLEAN_MAX_COLUMNS:
        active_chain = chain or _build_cleaning_chain()
        return _clean_column_values_with_ai(
            cleaned_df,
            ai_columns=ai_columns,
            user_prompt=user_prompt,
            chain=active_chain,
            ai_value_cache=ai_row_cache,
        )

    # Fill NAs to None for JSON
    df_clean = cleaned_df.astype(object).where(pd.notnull(cleaned_df), None)
    records = df_clean.to_dict(orient="records")
    ai_records = [_build_ai_view_row(record, ai_columns) for record in records]
    row_hashes = _compute_record_hashes(ai_records, columns=ai_columns)
    resolved_fragments: list[dict[str, Any] | None] = [None] * len(records)
    positions_by_hash: dict[int, list[int]] = {}
    unique_records: list[dict[str, Any]] = []
    unique_hashes: list[int] = []

    for index, (record, row_hash) in enumerate(zip(ai_records, row_hashes)):
        if ai_row_cache is not None and row_hash in ai_row_cache:
            resolved_fragments[index] = dict(ai_row_cache[row_hash])
            continue

        positions = positions_by_hash.setdefault(row_hash, [])
        positions.append(index)
        if len(positions) == 1:
            unique_records.append(record)
            unique_hashes.append(row_hash)

    for i in range(0, len(unique_records), batch_size):
        batch = unique_records[i:i + batch_size]
        batch_hashes = unique_hashes[i:i + batch_size]

        try:
            cleaned_batch = _invoke_cleaning_batch_with_fallback(
                chain=active_chain,
                batch_records=batch,
                user_prompt=user_prompt,
            )
        except Exception as e:
            print(f"Error cleaning batch {i} to {i + batch_size}: {e}")
            cleaned_batch = batch

        merged_batch = [
            _merge_cleaned_row(
                original_row=original_row,
                cleaned_row=cleaned_row,
                target_columns=target_columns,
            )
            for original_row, cleaned_row in zip(batch, cleaned_batch)
        ]

        for row_hash, merged_row in zip(batch_hashes, merged_batch):
            if ai_row_cache is not None:
                ai_row_cache[row_hash] = dict(merged_row)

            for row_index in positions_by_hash.get(row_hash, []):
                resolved_fragments[row_index] = dict(merged_row)

    final_rows = [
        _merge_cleaned_row(
            original_row=records[index],
            cleaned_row=resolved_fragments[index] if resolved_fragments[index] is not None else ai_records[index],
            target_columns=target_columns,
        )
        for index in range(len(records))
    ]
    return pd.DataFrame(final_rows, columns=cleaned_df.columns)
