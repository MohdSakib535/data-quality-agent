import re
from datetime import date, datetime
from typing import Any

import pandas as pd

from app.schemas.job import DataSuggestion

MAX_PROFILE_COLUMNS = 12
MAX_BAD_EXAMPLES = 3
MAX_SAMPLE_ROWS = 5

EMAIL_COLUMN_HINTS = ("email", "e-mail", "mail")
PHONE_COLUMN_HINTS = ("phone", "mobile", "cell", "contact", "whatsapp", "tel")
DATE_COLUMN_HINTS = ("date", "time", "dob", "birth", "timestamp")
NUMERIC_COLUMN_HINTS = ("amount", "price", "cost", "total", "qty", "quantity", "count", "score", "rate")
KEY_COLUMN_HINTS = ("id", "email", "phone", "mobile")

EMAIL_PATTERN = re.compile(r"^[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}$", re.IGNORECASE)
PHONE_PATTERN = re.compile(r"^\d{10,15}$")


def _to_snake_case(value: str) -> str:
    normalized = re.sub(r"[^0-9a-zA-Z]+", "_", value.strip())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized.lower()


def _sanitize_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _sanitize_json_value(item) for key, item in value.items()}

    if isinstance(value, list):
        return [_sanitize_json_value(item) for item in value]

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if pd.isna(value):
        return None

    if hasattr(value, "item"):
        return _sanitize_json_value(value.item())

    return value


def _append_issue(dataset_issues: list[str], issue: str) -> None:
    if issue not in dataset_issues:
        dataset_issues.append(issue)


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 2) if values else 0.0


def _series_to_string(series: pd.Series) -> pd.Series:
    return series.astype("string")


def _non_null_string_values(series: pd.Series) -> pd.Series:
    string_series = _series_to_string(series)
    return string_series[series.notna()]


def _bad_examples(series: pd.Series, mask: pd.Series) -> list[str]:
    examples = (
        _series_to_string(series)[mask]
        .dropna()
        .astype(str)
        .str.strip()
        .loc[lambda values: values.ne("")]
        .drop_duplicates()
        .head(MAX_BAD_EXAMPLES)
        .tolist()
    )
    return [str(example) for example in examples]


def _infer_column_type(series: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        return "number"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    return "string"


def _candidate_by_name(column_name: str, hints: tuple[str, ...]) -> bool:
    normalized = column_name.strip().lower()
    return any(hint in normalized for hint in hints)


def _compact_column_profile(column_profile: dict[str, Any]) -> dict[str, Any]:
    allowed_keys = (
        "name",
        "type",
        "null_percent",
        "invalid_phone_percent",
        "invalid_email_percent",
        "invalid_date_percent",
        "invalid_numeric_percent",
        "key_duplicate_percent",
        "casing_inconsistency_percent",
        "whitespace_inconsistency_percent",
        "examples",
    )
    compact = {key: column_profile[key] for key in allowed_keys if key in column_profile}
    return compact


def _column_issue_percent(column_profile: dict[str, Any]) -> float:
    issue_keys = (
        "null_percent",
        "invalid_phone_percent",
        "invalid_email_percent",
        "invalid_date_percent",
        "invalid_numeric_percent",
        "key_duplicate_percent",
        "casing_inconsistency_percent",
        "whitespace_inconsistency_percent",
    )
    return max(float(column_profile.get(key, 0.0)) for key in issue_keys)


def build_dataset_profile(df: pd.DataFrame) -> dict[str, Any]:
    row_count = len(df)
    row_count_safe = max(row_count, 1)
    duplicate_row_percent = round(float(df.duplicated().sum()) / row_count_safe * 100, 2) if row_count else 0.0

    dataset_issues: list[str] = []
    null_percents: list[float] = []
    invalid_format_percents: list[float] = []
    text_inconsistency_percents: list[float] = []
    numeric_parse_failure_percents: list[float] = []
    column_profiles: list[dict[str, Any]] = []

    sampled_rows = df.astype(object).where(pd.notnull(df), None).head(MAX_SAMPLE_ROWS)
    sample_rows = _sanitize_json_value(sampled_rows.to_dict(orient="records"))

    if duplicate_row_percent > 0:
        _append_issue(dataset_issues, "duplicate_rows")

    for column in df.columns:
        column_name = str(column)
        series = df[column]
        column_type = _infer_column_type(series)
        string_series = _series_to_string(series) if column_type == "string" else None
        stripped_series = string_series.str.strip() if string_series is not None else None
        blank_mask = stripped_series.eq("") if stripped_series is not None else pd.Series(False, index=series.index)
        null_mask = series.isna() | blank_mask
        null_percent = round(float(null_mask.mean()) * 100, 2) if row_count else 0.0
        null_percents.append(null_percent)

        profile: dict[str, Any] = {
            "name": column_name,
            "type": column_type,
            "null_percent": null_percent,
        }
        issue_score = null_percent

        if null_percent > 0:
            _append_issue(dataset_issues, "missing_values")

        non_null_mask = ~null_mask
        non_null_count = int(non_null_mask.sum())

        if column_type != "string" or non_null_count == 0:
            if _candidate_by_name(column_name, KEY_COLUMN_HINTS):
                non_null_values = series[non_null_mask]
                key_duplicate_percent = round(float(non_null_values.duplicated().mean()) * 100, 2) if not non_null_values.empty else 0.0
                if key_duplicate_percent > 0:
                    profile["key_duplicate_percent"] = key_duplicate_percent
                    issue_score = max(issue_score, key_duplicate_percent)
                    _append_issue(dataset_issues, "key_column_duplicates")

            column_profiles.append({**profile, "_issue_score": issue_score})
            continue

        stripped_non_null = stripped_series[non_null_mask]
        column_name_lower = column_name.lower()

        is_email_candidate = _candidate_by_name(column_name_lower, EMAIL_COLUMN_HINTS)
        email_validity_ratio = 0.0
        if not is_email_candidate:
            email_validity_ratio = float(stripped_non_null.str.fullmatch(EMAIL_PATTERN, na=False).mean()) if non_null_count else 0.0
            is_email_candidate = email_validity_ratio >= 0.6

        is_phone_candidate = _candidate_by_name(column_name_lower, PHONE_COLUMN_HINTS)
        phone_digits = stripped_non_null.str.replace(r"\D+", "", regex=True)
        phone_validity_ratio = float(phone_digits.str.fullmatch(PHONE_PATTERN, na=False).mean()) if non_null_count else 0.0
        if not is_phone_candidate:
            is_phone_candidate = phone_validity_ratio >= 0.6

        stripped_for_numeric = stripped_non_null.str.replace(r"[,\s$₹€£%]", "", regex=True)
        numeric_parsed = pd.to_numeric(stripped_for_numeric, errors="coerce")
        numeric_success_ratio = float(numeric_parsed.notna().mean()) if non_null_count else 0.0
        is_numeric_candidate = _candidate_by_name(column_name_lower, NUMERIC_COLUMN_HINTS)
        if not is_numeric_candidate and not is_phone_candidate and not is_email_candidate:
            is_numeric_candidate = numeric_success_ratio >= 0.6

        parsed_dates = pd.to_datetime(stripped_non_null, errors="coerce")
        date_success_ratio = float(parsed_dates.notna().mean()) if non_null_count else 0.0
        is_date_candidate = _candidate_by_name(column_name_lower, DATE_COLUMN_HINTS)
        if not is_date_candidate and not is_phone_candidate and not is_email_candidate:
            is_date_candidate = date_success_ratio >= 0.6

        if is_email_candidate:
            invalid_email_mask = non_null_mask & ~stripped_series.str.fullmatch(EMAIL_PATTERN, na=False)
            invalid_email_percent = round(float(invalid_email_mask.sum()) / row_count_safe * 100, 2)
            if invalid_email_percent > 0:
                profile["invalid_email_percent"] = invalid_email_percent
                profile["examples"] = _bad_examples(series, invalid_email_mask)
                invalid_format_percents.append(invalid_email_percent)
                issue_score = max(issue_score, invalid_email_percent)
                _append_issue(dataset_issues, "email_format_irregularities")

        if is_phone_candidate:
            phone_digits_full = _series_to_string(series).str.replace(r"\D+", "", regex=True)
            invalid_phone_mask = non_null_mask & ~phone_digits_full.str.fullmatch(PHONE_PATTERN, na=False)
            invalid_phone_percent = round(float(invalid_phone_mask.sum()) / row_count_safe * 100, 2)
            if invalid_phone_percent > 0:
                profile["invalid_phone_percent"] = invalid_phone_percent
                profile["examples"] = _bad_examples(series, invalid_phone_mask)
                invalid_format_percents.append(invalid_phone_percent)
                issue_score = max(issue_score, invalid_phone_percent)
                _append_issue(dataset_issues, "phone_format_irregularities")

        if is_date_candidate:
            parsed_dates_full = pd.to_datetime(stripped_series.where(non_null_mask), errors="coerce")
            invalid_date_mask = non_null_mask & parsed_dates_full.isna()
            invalid_date_percent = round(float(invalid_date_mask.sum()) / row_count_safe * 100, 2)
            if invalid_date_percent > 0:
                profile["invalid_date_percent"] = invalid_date_percent
                profile["examples"] = _bad_examples(series, invalid_date_mask)
                invalid_format_percents.append(invalid_date_percent)
                issue_score = max(issue_score, invalid_date_percent)
                _append_issue(dataset_issues, "mixed_date_formats")

        if is_numeric_candidate and not is_phone_candidate:
            numeric_clean_full = string_series.str.replace(r"[,\s$₹€£%]", "", regex=True)
            parsed_numeric_full = pd.to_numeric(numeric_clean_full.where(non_null_mask), errors="coerce")
            invalid_numeric_mask = non_null_mask & parsed_numeric_full.isna()
            invalid_numeric_percent = round(float(invalid_numeric_mask.sum()) / row_count_safe * 100, 2)
            if invalid_numeric_percent > 0:
                profile["invalid_numeric_percent"] = invalid_numeric_percent
                profile["examples"] = _bad_examples(series, invalid_numeric_mask)
                numeric_parse_failure_percents.append(invalid_numeric_percent)
                issue_score = max(issue_score, invalid_numeric_percent)
                _append_issue(dataset_issues, "numeric_parse_failures")

        if _candidate_by_name(column_name_lower, KEY_COLUMN_HINTS):
            normalized_key_values = stripped_non_null.str.lower()
            key_duplicate_percent = round(float(normalized_key_values.duplicated().mean()) * 100, 2) if not normalized_key_values.empty else 0.0
            if key_duplicate_percent > 0:
                profile["key_duplicate_percent"] = key_duplicate_percent
                issue_score = max(issue_score, key_duplicate_percent)
                _append_issue(dataset_issues, "key_column_duplicates")

        unique_count = int(stripped_non_null.nunique(dropna=True))
        unique_ratio = (unique_count / non_null_count) if non_null_count else 0.0
        average_length = float(stripped_non_null.str.len().fillna(0).mean()) if non_null_count else 0.0
        is_moderate_cardinality = unique_count <= 50 and unique_ratio <= 0.5 and average_length <= 80
        if is_moderate_cardinality and not any((is_email_candidate, is_phone_candidate, is_date_candidate, is_numeric_candidate)):
            whitespace_inconsistency_percent = round(float((string_series[non_null_mask] != stripped_non_null).mean()) * 100, 2)
            lower_unique_count = int(stripped_non_null.str.lower().nunique(dropna=True))
            casing_inconsistency_percent = 0.0
            if unique_count > 0 and lower_unique_count < unique_count:
                casing_inconsistency_percent = round(float(unique_count - lower_unique_count) / unique_count * 100, 2)

            text_issue_percent = max(whitespace_inconsistency_percent, casing_inconsistency_percent)
            if whitespace_inconsistency_percent > 0:
                profile["whitespace_inconsistency_percent"] = whitespace_inconsistency_percent
            if casing_inconsistency_percent > 0:
                profile["casing_inconsistency_percent"] = casing_inconsistency_percent
            if text_issue_percent > 0:
                text_inconsistency_percents.append(text_issue_percent)
                issue_score = max(issue_score, text_issue_percent)
                examples_mask = non_null_mask & (string_series != stripped_series)
                if not examples_mask.any() and casing_inconsistency_percent > 0:
                    duplicated_normalized = stripped_series.str.lower().duplicated(keep=False)
                    examples_mask = non_null_mask & duplicated_normalized
                if "examples" not in profile:
                    profile["examples"] = _bad_examples(series, examples_mask)
                _append_issue(dataset_issues, "text_normalization_needed")

        column_profiles.append({**profile, "_issue_score": issue_score})

    ordered_columns = sorted(column_profiles, key=lambda item: item.get("_issue_score", 0.0), reverse=True)
    compact_columns = [_compact_column_profile(column) for column in ordered_columns[:MAX_PROFILE_COLUMNS]]
    affected_column_count = sum(1 for column in column_profiles if column.get("_issue_score", 0.0) > 0.0)
    max_column_issue_percent = max((float(column.get("_issue_score", 0.0)) for column in column_profiles), default=0.0)

    return {
        "row_count_sampled": row_count,
        "total_columns": len(column_profiles),
        "affected_column_count": affected_column_count,
        "duplicate_row_percent": duplicate_row_percent,
        "affected_column_percent": round(float(affected_column_count) / max(len(column_profiles), 1) * 100, 2),
        "max_column_issue_percent": round(max_column_issue_percent, 2),
        "average_null_percent": _mean(null_percents),
        "average_invalid_format_percent": _mean(invalid_format_percents),
        "average_text_inconsistency_percent": _mean(text_inconsistency_percents),
        "average_numeric_parse_failure_percent": _mean(numeric_parse_failure_percents),
        "columns": compact_columns,
        "dataset_issues": dataset_issues,
        "sample_rows": sample_rows,
    }


def compute_quality_score(profile: dict[str, Any]) -> int:
    columns = profile.get("columns", [])
    total_columns = max(int(profile.get("total_columns", len(columns) or 1)), 1)
    affected_column_count = int(
        profile.get(
            "affected_column_count",
            sum(1 for column in columns if _column_issue_percent(column) > 0.0),
        )
    )
    affected_column_percent = float(
        profile.get("affected_column_percent", (affected_column_count / total_columns) * 100.0)
    )
    max_column_issue_percent = float(
        profile.get(
            "max_column_issue_percent",
            max((_column_issue_percent(column) for column in columns), default=0.0),
        )
    )
    dataset_issue_count = len(profile.get("dataset_issues", []))

    penalty = 0.0
    penalty += min(20.0, 20.0 * (float(profile.get("average_null_percent", 0.0)) / 100.0))
    penalty += min(20.0, 20.0 * (float(profile.get("average_invalid_format_percent", 0.0)) / 100.0))
    penalty += min(10.0, 10.0 * (float(profile.get("duplicate_row_percent", 0.0)) / 100.0))
    penalty += min(10.0, 10.0 * (float(profile.get("average_text_inconsistency_percent", 0.0)) / 100.0))
    penalty += min(10.0, 10.0 * (float(profile.get("average_numeric_parse_failure_percent", 0.0)) / 100.0))
    penalty += min(10.0, 10.0 * (affected_column_percent / 100.0))
    penalty += min(15.0, 15.0 * (max_column_issue_percent / 100.0))
    penalty += min(5.0, 5.0 * (dataset_issue_count / 6.0))

    return max(0, min(100, int(round(100.0 - penalty))))


def generate_rule_based_suggestions(profile: dict[str, Any]) -> list[DataSuggestion]:
    suggestions: list[DataSuggestion] = []
    columns = profile.get("columns", [])
    duplicate_row_percent = float(profile.get("duplicate_row_percent", 0.0))

    if any(float(column.get("invalid_phone_percent", 0.0)) > 5.0 for column in columns):
        suggestions.append(
            DataSuggestion(
                issue_description="Phone-like columns contain invalid or inconsistent formats in the sampled data.",
                priority="High",
                resolution_prompt=(
                    "Standardize phone values into one consistent format, keep only valid phone-like records, "
                    "remove punctuation noise, preserve country codes when present, and avoid changing unrelated text columns."
                ),
            )
        )

    if any(float(column.get("invalid_date_percent", 0.0)) > 0.0 for column in columns):
        suggestions.append(
            DataSuggestion(
                issue_description="Date-like columns show mixed parsing success, indicating inconsistent date formats.",
                priority="High",
                resolution_prompt=(
                    "Normalize date-like values into one consistent date format across the dataset, preserve valid dates, "
                    "leave non-date columns untouched, and flag or standardize entries that do not parse cleanly."
                ),
            )
        )

    if any(float(column.get("null_percent", 0.0)) > 10.0 for column in columns):
        suggestions.append(
            DataSuggestion(
                issue_description="One or more columns contain a meaningful share of missing or blank values.",
                priority="Medium",
                resolution_prompt=(
                    "Review missing and blank values across affected columns, normalize empty placeholders consistently, "
                    "replace only genuinely missing entries with the chosen placeholder, and do not invent new data."
                ),
            )
        )

    if duplicate_row_percent > 1.0:
        suggestions.append(
            DataSuggestion(
                issue_description="The sampled dataset contains exact duplicate rows.",
                priority="High",
                resolution_prompt=(
                    "Identify exact duplicate records and remove redundant copies while preserving one canonical row for each duplicate set. "
                    "Do not perform fuzzy matching or merge records that are not exact duplicates."
                ),
            )
        )

    if any(
        float(column.get("casing_inconsistency_percent", 0.0)) > 0.0
        or float(column.get("whitespace_inconsistency_percent", 0.0)) > 0.0
        for column in columns
    ):
        suggestions.append(
            DataSuggestion(
                issue_description="Some moderate-cardinality text columns show casing or whitespace inconsistencies.",
                priority="Medium",
                resolution_prompt=(
                    "Trim leading and trailing whitespace and standardize casing for repeated text values that represent the same meaning. "
                    "Preserve free-text content and do not rewrite unrelated text."
                ),
            )
        )

    if any(float(column.get("invalid_numeric_percent", 0.0)) > 5.0 for column in columns):
        suggestions.append(
            DataSuggestion(
                issue_description="Some numeric-like columns contain values that do not parse cleanly as numbers.",
                priority="Medium",
                resolution_prompt=(
                    "Normalize numeric-like fields by removing formatting noise such as currency symbols and separators where appropriate, "
                    "preserve valid numeric values, and isolate entries that still fail numeric parsing after normalization."
                ),
            )
        )

    return suggestions[:5]
