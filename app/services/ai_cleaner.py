import json
import re
import ast
from typing import List, Dict, Any
import pandas as pd
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from app.prompts.cleaning_prompts import analysis_prompt_template, clean_data_prompt_template
from app.schemas.job import DataSuggestion, DatasetAnalysisPayload, DatasetAnalysisResponse
from app.core.config import settings
from app.services.deterministic_cleaner import (
    _to_snake_case,
    analyze_dataset_deterministically,
)

try:
    from langchain_ollama import ChatOllama
except ImportError:
    from langchain_community.chat_models import ChatOllama


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

def _suggestion_key(suggestion: DataSuggestion) -> str:
    return "|".join(
        [
            suggestion.issue_description.strip().lower(),
            suggestion.priority.strip().lower(),
            suggestion.resolution_prompt.strip().lower(),
        ]
    )


def _generic_resolution_prompt(issue_description: str, current_prompt: str) -> str:
    suggestion_text = f"{issue_description} {current_prompt}".lower()

    if any(keyword in suggestion_text for keyword in ["missing", "null", "blank", "empty", "n/a"]):
        return (
            "Review the dataset for missing or placeholder values across all affected columns and replace them "
            "with the literal string N/A wherever the value is missing. Treat blank strings, whitespace-only "
            "cells, null, NULL, NA, -, unknown, and other visibly empty placeholders as missing values. Apply "
            "the replacement consistently across the dataset, preserve all non-empty valid values as they are, "
            "and do not invent or guess any new data beyond converting missing entries to N/A."
        )

    if "duplicate" in suggestion_text:
        return (
            "Identify exact duplicate records across the dataset and remove redundant copies while keeping one "
            "canonical version of each repeated row. Preserve legitimate repeated values that are not true "
            "duplicates, and avoid merging rows unless all corresponding fields clearly represent the same record."
        )

    if any(keyword in suggestion_text for keyword in ["header", "column name", "column names", "snake_case"]):
        return (
            "Normalize all column headers consistently across the dataset. Trim surrounding whitespace, remove "
            "punctuation noise, convert names to lowercase snake_case, and ensure each header is clear, stable, "
            "and unique without changing the meaning of the column."
        )

    if "whitespace" in suggestion_text or "leading or trailing" in suggestion_text:
        return (
            "Trim leading and trailing whitespace in all affected text fields across the dataset. Preserve "
            "meaningful internal spacing unless it is clearly accidental, and ensure values that become empty "
            "after trimming are handled consistently as blank or missing data."
        )

    if any(keyword in suggestion_text for keyword in ["capitalization", "casing", "spelling", "formatting"]):
        return (
            "Standardize capitalization, spelling, and formatting for repeated text values that represent the same "
            "meaning. Apply one canonical form consistently across the dataset while preserving genuinely distinct "
            "values and avoiding unsupported corrections."
        )

    return (
        f"Apply a consistent dataset-wide cleaning rule for this issue: {issue_description.strip()} "
        "Review all affected rows and columns, standardize equivalent values into one canonical format, preserve "
        "valid distinctions, and avoid guessing or changing unrelated data."
    )


def _normalize_analysis_suggestion(suggestion: DataSuggestion) -> DataSuggestion:
    return DataSuggestion(
        issue_description=suggestion.issue_description.strip(),
        priority=suggestion.priority.strip(),
        resolution_prompt=_generic_resolution_prompt(
            suggestion.issue_description,
            suggestion.resolution_prompt,
        ),
    )


def _merge_analysis_payloads(
    deterministic: DatasetAnalysisPayload,
    ai_payload: DatasetAnalysisPayload | None,
) -> DatasetAnalysisPayload:
    deterministic = DatasetAnalysisPayload(
        quality_score=deterministic.quality_score,
        suggestions=[_normalize_analysis_suggestion(s) for s in deterministic.suggestions],
    )

    if ai_payload is None:
        return deterministic

    ai_payload = DatasetAnalysisPayload(
        quality_score=ai_payload.quality_score,
        suggestions=[_normalize_analysis_suggestion(s) for s in ai_payload.suggestions],
    )

    merged_suggestions: List[DataSuggestion] = []
    seen = set()

    for suggestion in list(ai_payload.suggestions) + list(deterministic.suggestions):
        key = _suggestion_key(suggestion)
        if key in seen:
            continue
        seen.add(key)
        merged_suggestions.append(suggestion)

    merged_score = int(round((deterministic.quality_score + ai_payload.quality_score) / 2))
    return DatasetAnalysisPayload(
        quality_score=max(0, min(100, merged_score)),
        suggestions=merged_suggestions[:5],
    )


def _build_dataset_profile(df: pd.DataFrame) -> Dict[str, Any]:
    row_count = len(df)
    column_count = len(df.columns)
    total_cells = max(row_count * max(column_count, 1), 1)
    missing_cells = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum()) if row_count > 0 else 0

    dtype_counts: Dict[str, int] = {}
    for dtype in df.dtypes.astype(str):
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1

    object_columns = df.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()

    missing_by_column = (
        df.isna()
        .mean()
        .sort_values(ascending=False)
        .head(8)
    )
    highest_missing_columns = [
        {
            "column": str(column),
            "missing_ratio": round(float(ratio), 4),
        }
        for column, ratio in missing_by_column.items()
        if float(ratio) > 0
    ]

    text_column_samples = []
    for column in object_columns[:8]:
        series = df[column].dropna().astype(str).head(3).tolist()
        if series:
            text_column_samples.append({
                "column": str(column),
                "examples": series,
            })

    numeric_column_ranges = []
    for column in numeric_columns[:8]:
        series = df[column].dropna()
        if not series.empty:
            numeric_column_ranges.append({
                "column": str(column),
                "min": float(series.min()),
                "max": float(series.max()),
            })

    return {
        "row_count": row_count,
        "column_count": column_count,
        "missing_cell_ratio": round(missing_cells / total_cells, 4),
        "duplicate_row_ratio": round(duplicate_rows / max(row_count, 1), 4),
        "dtype_counts": dtype_counts,
        "highest_missing_columns": highest_missing_columns,
        "text_column_samples": text_column_samples,
        "numeric_column_ranges": numeric_column_ranges,
    }


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
        or _should_normalize_headers(user_prompt)
        or _should_trim_whitespace(user_prompt)
    )

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

    return not deterministic_only


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

        parsed = pd.to_datetime(
            series_non_null,
            errors="coerce",
            infer_datetime_format=True,
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


def analyze_dataset(df: pd.DataFrame) -> DatasetAnalysisResponse:
    """
    Sample the dataframe and ask the AI to analyze its quality and suggest cleaning prompts.
    """
    deterministic_analysis = analyze_dataset_deterministically(df)

    # Scale row sampling by column count so large wide datasets stay within context.
    target_cell_budget = 1200
    sample_size = min(
        len(df),
        max(5, min(20, target_cell_budget // max(len(df.columns), 1))),
    )
    sample_df = df.sample(n=sample_size, random_state=42) if len(df) > 50 else df
    dataset_profile_json = json.dumps(_build_dataset_profile(df), indent=2)
    
    # Fill NA values with None for JSON serialization
    sample_df = sample_df.where(pd.notnull(sample_df), None)
    dataset_sample_json = sample_df.to_json(orient="records", date_format="iso")
    
    llm = ChatOllama(model=settings.OLLAMA_MODEL, base_url=settings.OLLAMA_BASE_URL, temperature=0.0)
    parser = JsonOutputParser(pydantic_object=DatasetAnalysisPayload)
    
    chain = analysis_prompt_template | llm | StrOutputParser()
    
    try:
        raw_response = chain.invoke({
            "dataset_profile_json": dataset_profile_json,
            "dataset_sample_json": dataset_sample_json,
            "format_instructions": parser.get_format_instructions()
        })
        
        result_dict = _extract_json_payload(raw_response)
        
        # In case the payload is wrapped under a key
        if isinstance(result_dict, dict) and "suggestions" not in result_dict and len(result_dict) == 1:
             result_dict = list(result_dict.values())[0]

        ai_payload = DatasetAnalysisPayload(**result_dict)
        merged_payload = _merge_analysis_payloads(deterministic_analysis, ai_payload)
        return DatasetAnalysisResponse(
            job_id="",
            quality_score=merged_payload.quality_score,
            suggestions=merged_payload.suggestions,
        )
    except Exception as e:
        print(f"AI Analysis failed: {e}")
        # Fall back to deterministic profiling instead of returning an empty analysis.
        return DatasetAnalysisResponse(
            job_id="",
            quality_score=deterministic_analysis.quality_score,
            suggestions=deterministic_analysis.suggestions,
        )

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
