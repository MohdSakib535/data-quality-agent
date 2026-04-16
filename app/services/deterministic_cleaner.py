import re
import warnings
from datetime import date, datetime
from numbers import Integral, Real
from typing import Any, Iterable

import pandas as pd

from app.core.config import settings
from app.schemas.job import DataSuggestion

MAX_PROFILE_COLUMNS = 12
MAX_BAD_EXAMPLES = 3
MAX_SAMPLE_ROWS = 5

EMAIL_COLUMN_HINTS = ("email", "e-mail", "mail")
PHONE_COLUMN_HINTS = ("phone", "mobile", "cell", "contact", "whatsapp", "tel")
DATE_COLUMN_HINTS = ("date", "time", "dob", "birth", "timestamp")
AGE_COLUMN_HINTS = ("age", "age_years", "years_old")
BOOLEAN_COLUMN_HINTS = ("is", "has", "flag", "enabled", "disabled", "active", "inactive", "verified", "approved")
INTEGER_COLUMN_HINTS = AGE_COLUMN_HINTS + ("count", "qty", "quantity", "rank", "index", "position", "number")
FLOAT_COLUMN_HINTS = ("amount", "price", "cost", "total", "rate", "ratio", "score", "percent", "percentage", "balance")
NUMERIC_COLUMN_HINTS = ("amount", "price", "cost", "total", "qty", "quantity", "count", "score", "rate")
TEXT_SEMANTIC_COLUMN_HINTS = (
    "name",
    "city",
    "state",
    "country",
    "title",
    "role",
    "department",
    "designation",
    "category",
    "status",
    "company",
    "skill",
    "profession",
)
KEY_COLUMN_HINTS = ("id", "email", "phone", "mobile")

EMAIL_PATTERN = re.compile(r"^[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}$", re.IGNORECASE)
PHONE_PATTERN = re.compile(r"^\d{10,15}$")
LETTER_PATTERN = re.compile(r"[A-Za-z]")
AGE_SUFFIX_PATTERN = re.compile(r"\b(years?\s*old|years?|yrs?|yo)\b", re.IGNORECASE)
AGE_MIN_VALUE = 0
AGE_MAX_VALUE = 120
AGE_NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
    "hundred": 100,
}
BOOLEAN_TRUE_VALUES = {"true", "1"}
BOOLEAN_FALSE_VALUES = {"false", "0"}


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


def _normalize_phone_for_profile(value: str) -> str:
    text = str(value).strip()
    digits = re.sub(r"\D+", "", text)
    if not digits or not PHONE_PATTERN.fullmatch(digits):
        return text

    # Treat digit-only values as already normalized, even when they contain a
    # country code without a leading "+".
    return digits


def _format_numeric_value(value: float) -> str:
    if pd.isna(value):
        return ""
    if float(value).is_integer():
        return str(int(float(value)))
    return format(float(value), "f").rstrip("0").rstrip(".")


def _looks_textual_value(value: str) -> bool:
    return bool(LETTER_PATTERN.search(str(value).strip()))


def _is_age_column_name(column_name: str) -> bool:
    normalized = _to_snake_case(column_name)
    tokens = [token for token in normalized.split("_") if token]
    token_set = set(tokens)
    return (
        "age" in token_set
        or normalized in AGE_COLUMN_HINTS
        or ("years" in token_set and "old" in token_set)
    )


def _is_integer_column_name(column_name: str) -> bool:
    normalized = _to_snake_case(column_name)
    tokens = [token for token in normalized.split("_") if token]
    token_set = set(tokens)
    return bool(token_set & set(INTEGER_COLUMN_HINTS)) or normalized in INTEGER_COLUMN_HINTS


def _is_float_column_name(column_name: str) -> bool:
    normalized = _to_snake_case(column_name)
    tokens = [token for token in normalized.split("_") if token]
    token_set = set(tokens)
    return bool(token_set & set(FLOAT_COLUMN_HINTS)) or normalized in FLOAT_COLUMN_HINTS


def _is_boolean_column_name(column_name: str) -> bool:
    normalized = _to_snake_case(column_name)
    tokens = [token for token in normalized.split("_") if token]
    token_set = set(tokens)
    if normalized.startswith("is_") or normalized.startswith("has_"):
        return True
    return bool(token_set & set(BOOLEAN_COLUMN_HINTS)) or normalized in BOOLEAN_COLUMN_HINTS


def _parse_number_words(value: str) -> int | None:
    normalized = AGE_SUFFIX_PATTERN.sub(" ", value.lower())
    normalized = re.sub(r"[^a-z\s-]", " ", normalized)
    normalized = normalized.replace("-", " ")
    tokens = [token for token in normalized.split() if token and token != "and"]
    if not tokens:
        return None

    total = 0
    current = 0
    for token in tokens:
        if token not in AGE_NUMBER_WORDS:
            return None
        number = AGE_NUMBER_WORDS[token]
        if token == "hundred":
            current = max(1, current) * number
        else:
            current += number
    total += current
    return total


def _parse_strict_integer_value(
    value: Any,
    *,
    min_value: int | None = None,
    max_value: int | None = None,
) -> int | None:
    if pd.isna(value) or isinstance(value, bool):
        return None

    parsed_value: int | None = None
    if isinstance(value, Integral):
        parsed_value = int(value)
    elif isinstance(value, Real):
        numeric_value = float(value)
        if not numeric_value.is_integer():
            return None
        parsed_value = int(numeric_value)
    else:
        return None

    if min_value is not None and parsed_value < min_value:
        return None
    if max_value is not None and parsed_value > max_value:
        return None
    return parsed_value


def _parse_float_value(value: Any) -> float | None:
    if pd.isna(value) or isinstance(value, bool):
        return None

    if isinstance(value, Real):
        return float(value)

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    return None


def _parse_boolean_value(value: Any) -> bool | None:
    if pd.isna(value):
        return None

    if isinstance(value, bool):
        return value

    if isinstance(value, Integral):
        if int(value) == 1:
            return True
        if int(value) == 0:
            return False
        return None

    if isinstance(value, Real):
        numeric_value = float(value)
        if numeric_value == 1.0:
            return True
        if numeric_value == 0.0:
            return False
        return None

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in BOOLEAN_TRUE_VALUES:
            return True
        if normalized in BOOLEAN_FALSE_VALUES:
            return False
    return None


def _parse_age_value(value: Any) -> int | None:
    return _parse_strict_integer_value(
        value,
        min_value=AGE_MIN_VALUE,
        max_value=AGE_MAX_VALUE,
    )


def _parse_datetime_series(values: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(
            values,
            errors="coerce",
            format="mixed",
            utc=False,
        )
    except (TypeError, ValueError):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Could not infer format, so each element will be parsed individually, falling back to `dateutil`\..*",
                category=UserWarning,
            )
            return pd.to_datetime(
                values,
                errors="coerce",
                utc=False,
            )


def _compact_column_profile(column_profile: dict[str, Any]) -> dict[str, Any]:
    allowed_keys = (
        "name",
        "type",
        "expected_type",
        "null_percent",
        "invalid_boolean_percent",
        "invalid_integer_percent",
        "invalid_float_percent",
        "invalid_phone_percent",
        "phone_format_inconsistency_percent",
        "invalid_email_percent",
        "invalid_date_percent",
        "date_format_inconsistency_percent",
        "invalid_numeric_percent",
        "numeric_format_inconsistency_percent",
        "invalid_age_percent",
        "age_format_inconsistency_percent",
        "semantic_type_mismatch_percent",
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
        "invalid_boolean_percent",
        "invalid_integer_percent",
        "invalid_float_percent",
        "invalid_phone_percent",
        "phone_format_inconsistency_percent",
        "invalid_email_percent",
        "invalid_date_percent",
        "date_format_inconsistency_percent",
        "invalid_numeric_percent",
        "numeric_format_inconsistency_percent",
        "invalid_age_percent",
        "age_format_inconsistency_percent",
        "semantic_type_mismatch_percent",
        "key_duplicate_percent",
        "casing_inconsistency_percent",
        "whitespace_inconsistency_percent",
    )
    return max(float(column_profile.get(key, 0.0)) for key in issue_keys)


def _append_examples(examples: list[str], values: Iterable[Any]) -> None:
    for value in values:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if not text or text in examples:
            continue
        examples.append(text)
        if len(examples) >= MAX_BAD_EXAMPLES:
            return


def _init_column_stats(column_name: str) -> dict[str, Any]:
    return {
        "name": column_name,
        "saw_string": False,
        "saw_number": False,
        "saw_boolean": False,
        "null_count": 0,
        "non_null_count": 0,
        "email_valid_count": 0,
        "phone_valid_count": 0,
        "phone_format_inconsistency_count": 0,
        "boolean_valid_count": 0,
        "integer_valid_count": 0,
        "float_valid_count": 0,
        "numeric_valid_count": 0,
        "numeric_format_inconsistency_count": 0,
        "age_valid_count": 0,
        "age_format_inconsistency_count": 0,
        "date_valid_count": 0,
        "date_format_inconsistency_count": 0,
        "text_like_count": 0,
        "key_duplicate_count": 0,
        "whitespace_inconsistency_count": 0,
        "text_length_sum": 0,
        "unique_values": set(),
        "lower_unique_values": set(),
        "track_uniques": True,
        "invalid_email_examples": [],
        "invalid_phone_examples": [],
        "phone_format_examples": [],
        "invalid_boolean_examples": [],
        "invalid_integer_examples": [],
        "invalid_float_examples": [],
        "invalid_date_examples": [],
        "date_format_examples": [],
        "invalid_numeric_examples": [],
        "numeric_format_examples": [],
        "invalid_age_examples": [],
        "age_format_examples": [],
        "semantic_type_mismatch_examples": [],
        "text_examples": [],
        "seen_key_values": set(),
    }


def _infer_aggregated_column_type(stats: dict[str, Any]) -> str:
    if stats["saw_string"]:
        return "string"
    if stats["saw_number"] and not stats["saw_boolean"]:
        return "number"
    if stats["saw_boolean"] and not stats["saw_number"]:
        return "boolean"
    return "string"


def _infer_expected_validation_type(column_name: str, stats: dict[str, Any], non_null_count: int) -> str:
    if non_null_count <= 0:
        return "string"

    normalized_name = column_name.lower()
    if _candidate_by_name(normalized_name, EMAIL_COLUMN_HINTS) or _candidate_by_name(normalized_name, PHONE_COLUMN_HINTS):
        return "string"

    boolean_ratio = float(stats["boolean_valid_count"]) / non_null_count
    integer_ratio = float(stats["integer_valid_count"]) / non_null_count
    float_ratio = float(stats["float_valid_count"]) / non_null_count
    date_ratio = float(stats["date_valid_count"]) / non_null_count

    if _is_boolean_column_name(column_name) or boolean_ratio >= 0.6:
        return "boolean"
    if _candidate_by_name(normalized_name, DATE_COLUMN_HINTS) or date_ratio >= 0.6:
        return "date"
    if _is_age_column_name(column_name) or _is_integer_column_name(column_name) or integer_ratio >= 0.6:
        return "integer"
    if _is_float_column_name(column_name) or _candidate_by_name(normalized_name, NUMERIC_COLUMN_HINTS) or float_ratio >= 0.6:
        return "float"
    return "string"


def _update_column_stats(stats: dict[str, Any], series: pd.Series) -> None:
    if not pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        stats["saw_string"] = True
    if pd.api.types.is_numeric_dtype(series):
        stats["saw_number"] = True
    if pd.api.types.is_bool_dtype(series):
        stats["saw_boolean"] = True

    string_series = _series_to_string(series)
    stripped_series = string_series.str.strip()
    blank_mask = stripped_series.eq("")
    null_mask = series.isna() | blank_mask
    non_null_mask = ~null_mask
    non_null_count = int(non_null_mask.sum())

    stats["null_count"] += int(null_mask.sum())
    stats["non_null_count"] += non_null_count
    if non_null_count == 0:
        return

    stripped_non_null = stripped_series[non_null_mask]
    email_matches = stripped_non_null.str.fullmatch(EMAIL_PATTERN, na=False)
    stats["email_valid_count"] += int(email_matches.sum())
    _append_examples(stats["invalid_email_examples"], stripped_non_null[~email_matches].tolist())

    phone_digits = stripped_non_null.str.replace(r"\D+", "", regex=True)
    phone_matches = phone_digits.str.fullmatch(PHONE_PATTERN, na=False)
    stats["phone_valid_count"] += int(phone_matches.sum())
    _append_examples(stats["invalid_phone_examples"], stripped_non_null[~phone_matches].tolist())
    valid_phone_values = stripped_non_null[phone_matches]
    if not valid_phone_values.empty:
        normalized_phone_values = valid_phone_values.map(_normalize_phone_for_profile)
        phone_format_changes = normalized_phone_values.ne(valid_phone_values)
        stats["phone_format_inconsistency_count"] += int(phone_format_changes.sum())
        _append_examples(
            stats["phone_format_examples"],
            valid_phone_values[phone_format_changes].tolist(),
        )

    parsed_booleans = stripped_non_null.map(_parse_boolean_value)
    boolean_matches = parsed_booleans.notna()
    stats["boolean_valid_count"] += int(boolean_matches.sum())
    _append_examples(stats["invalid_boolean_examples"], stripped_non_null[~boolean_matches].tolist())

    parsed_integers = stripped_non_null.map(_parse_strict_integer_value)
    integer_matches = parsed_integers.notna()
    stats["integer_valid_count"] += int(integer_matches.sum())
    _append_examples(stats["invalid_integer_examples"], stripped_non_null[~integer_matches].tolist())

    parsed_floats = stripped_non_null.map(_parse_float_value)
    float_matches = parsed_floats.notna()
    stats["float_valid_count"] += int(float_matches.sum())
    _append_examples(stats["invalid_float_examples"], stripped_non_null[~float_matches].tolist())

    numeric_clean = stripped_non_null.str.replace(r"[,\s$₹€£%]", "", regex=True)
    numeric_parsed = pd.to_numeric(numeric_clean, errors="coerce")
    numeric_matches = numeric_parsed.notna()
    stats["numeric_valid_count"] += int(numeric_matches.sum())
    _append_examples(stats["invalid_numeric_examples"], stripped_non_null[~numeric_matches].tolist())
    valid_numeric_values = stripped_non_null[numeric_matches]
    if not valid_numeric_values.empty:
        formatted_numeric_values = numeric_parsed[numeric_matches].map(_format_numeric_value).astype("string")
        numeric_format_changes = formatted_numeric_values.ne(valid_numeric_values)
        stats["numeric_format_inconsistency_count"] += int(numeric_format_changes.sum())
        _append_examples(
            stats["numeric_format_examples"],
            valid_numeric_values[numeric_format_changes].tolist(),
        )

    parsed_ages = stripped_non_null.map(_parse_age_value)
    age_matches = parsed_ages.notna()
    stats["age_valid_count"] += int(age_matches.sum())
    _append_examples(stats["invalid_age_examples"], stripped_non_null[~age_matches].tolist())
    valid_age_values = stripped_non_null[age_matches]
    if not valid_age_values.empty:
        formatted_age_values = parsed_ages[age_matches].astype(int).astype("string")
        age_format_changes = formatted_age_values.ne(valid_age_values)
        stats["age_format_inconsistency_count"] += int(age_format_changes.sum())
        _append_examples(
            stats["age_format_examples"],
            valid_age_values[age_format_changes].tolist(),
        )

    parsed_dates = _parse_datetime_series(stripped_non_null)
    date_matches = parsed_dates.notna()
    stats["date_valid_count"] += int(date_matches.sum())
    _append_examples(stats["invalid_date_examples"], stripped_non_null[~date_matches].tolist())
    valid_date_values = stripped_non_null[date_matches]
    if not valid_date_values.empty:
        formatted_date_values = parsed_dates[date_matches].dt.strftime(settings.DATE_OUTPUT_FORMAT).astype("string")
        date_format_changes = formatted_date_values.ne(valid_date_values)
        stats["date_format_inconsistency_count"] += int(date_format_changes.sum())
        _append_examples(
            stats["date_format_examples"],
            valid_date_values[date_format_changes].tolist(),
        )

    stats["whitespace_inconsistency_count"] += int((string_series[non_null_mask] != stripped_non_null).sum())
    stats["text_length_sum"] += int(stripped_non_null.str.len().fillna(0).sum())
    text_like_matches = stripped_non_null.str.contains(LETTER_PATTERN, na=False)
    stats["text_like_count"] += int(text_like_matches.sum())
    _append_examples(
        stats["text_examples"],
        stripped_non_null[string_series[non_null_mask] != stripped_non_null].tolist(),
    )
    _append_examples(
        stats["semantic_type_mismatch_examples"],
        stripped_non_null[~text_like_matches].tolist(),
    )

    if stats["track_uniques"]:
        for value in stripped_non_null.tolist():
            text = str(value)
            stats["unique_values"].add(text)
            stats["lower_unique_values"].add(text.lower())
            if len(stats["unique_values"]) > 60:
                stats["track_uniques"] = False
                stats["unique_values"].clear()
                stats["lower_unique_values"].clear()
                break

    if _candidate_by_name(stats["name"], KEY_COLUMN_HINTS):
        for value in stripped_non_null.str.lower().tolist():
            if value in stats["seen_key_values"]:
                stats["key_duplicate_count"] += 1
            else:
                stats["seen_key_values"].add(value)


def build_dataset_profile_from_chunks(chunks: Iterable[pd.DataFrame]) -> dict[str, Any]:
    row_count = 0
    duplicate_row_count = 0
    dataset_issues: list[str] = []
    sample_rows: list[dict[str, Any]] = []
    column_stats: dict[str, dict[str, Any]] = {}
    column_order: list[str] = []
    seen_row_hashes: set[int] = set()

    for chunk_df in chunks:
        if chunk_df is None:
            continue

        chunk_len = len(chunk_df)
        row_count += chunk_len

        if not column_order:
            column_order = [str(column) for column in chunk_df.columns]
            for column_name in column_order:
                column_stats[column_name] = _init_column_stats(column_name)

        if len(sample_rows) < MAX_SAMPLE_ROWS and chunk_len:
            sample_take = min(MAX_SAMPLE_ROWS - len(sample_rows), chunk_len)
            sampled_rows = chunk_df.head(sample_take).astype(object).where(pd.notnull(chunk_df.head(sample_take)), None)
            sample_rows.extend(_sanitize_json_value(sampled_rows.to_dict(orient="records")))

        if chunk_len:
            comparable_df = chunk_df.astype(object).where(pd.notnull(chunk_df), None)
            row_hashes = (
                pd.util.hash_pandas_object(comparable_df.astype(str), index=False)
                .astype("uint64")
                .tolist()
            )
            local_seen: set[int] = set()
            for row_hash in row_hashes:
                if row_hash in seen_row_hashes or row_hash in local_seen:
                    duplicate_row_count += 1
                    continue
                local_seen.add(row_hash)
            seen_row_hashes.update(local_seen)

        for column in chunk_df.columns:
            column_name = str(column)
            if column_name not in column_stats:
                column_order.append(column_name)
                column_stats[column_name] = _init_column_stats(column_name)
            _update_column_stats(column_stats[column_name], chunk_df[column])

    row_count_safe = max(row_count, 1)
    duplicate_row_percent = round(float(duplicate_row_count) / row_count_safe * 100, 2) if row_count else 0.0
    if duplicate_row_percent > 0:
        _append_issue(dataset_issues, "duplicate_rows")

    null_percents: list[float] = []
    invalid_format_percents: list[float] = []
    text_inconsistency_percents: list[float] = []
    format_inconsistency_percents: list[float] = []
    numeric_parse_failure_percents: list[float] = []
    column_profiles: list[dict[str, Any]] = []

    for column_name in column_order:
        stats = column_stats[column_name]
        non_null_count = stats["non_null_count"]
        null_percent = round(float(stats["null_count"]) / row_count_safe * 100, 2) if row_count else 0.0
        null_percents.append(null_percent)

        profile: dict[str, Any] = {
            "name": column_name,
            "type": _infer_aggregated_column_type(stats),
            "null_percent": null_percent,
        }
        issue_score = null_percent

        if null_percent > 0:
            _append_issue(dataset_issues, "missing_values")

        if non_null_count == 0:
            if _candidate_by_name(column_name, KEY_COLUMN_HINTS):
                key_duplicate_percent = 0.0
                if key_duplicate_percent > 0:
                    profile["key_duplicate_percent"] = key_duplicate_percent
            column_profiles.append({**profile, "_issue_score": issue_score})
            continue

        column_name_lower = column_name.lower()
        email_validity_ratio = float(stats["email_valid_count"]) / non_null_count
        is_email_candidate = _candidate_by_name(column_name_lower, EMAIL_COLUMN_HINTS) or email_validity_ratio >= 0.6

        phone_validity_ratio = float(stats["phone_valid_count"]) / non_null_count
        is_phone_candidate = _candidate_by_name(column_name_lower, PHONE_COLUMN_HINTS) or phone_validity_ratio >= 0.6

        expected_type = _infer_expected_validation_type(column_name, stats, non_null_count)
        profile["expected_type"] = expected_type

        is_age_candidate = _is_age_column_name(column_name)
        numeric_success_ratio = float(stats["numeric_valid_count"]) / non_null_count
        is_numeric_candidate = _candidate_by_name(column_name_lower, NUMERIC_COLUMN_HINTS)
        if not is_numeric_candidate and not is_phone_candidate and not is_email_candidate and not is_age_candidate:
            is_numeric_candidate = numeric_success_ratio >= 0.6

        date_success_ratio = float(stats["date_valid_count"]) / non_null_count
        is_date_candidate = _candidate_by_name(column_name_lower, DATE_COLUMN_HINTS)
        if not is_date_candidate and not is_phone_candidate and not is_email_candidate:
            is_date_candidate = date_success_ratio >= 0.6

        if expected_type == "boolean":
            invalid_boolean_percent = round(
                float(non_null_count - stats["boolean_valid_count"]) / row_count_safe * 100,
                2,
            )
            if invalid_boolean_percent > 0:
                profile["invalid_boolean_percent"] = invalid_boolean_percent
                profile["examples"] = stats["invalid_boolean_examples"][:MAX_BAD_EXAMPLES]
                invalid_format_percents.append(invalid_boolean_percent)
                issue_score = max(issue_score, invalid_boolean_percent)
                _append_issue(dataset_issues, "boolean_validation_needed")

        if is_email_candidate:
            invalid_email_percent = round(float(non_null_count - stats["email_valid_count"]) / row_count_safe * 100, 2)
            if invalid_email_percent > 0:
                profile["invalid_email_percent"] = invalid_email_percent
                profile["examples"] = stats["invalid_email_examples"][:MAX_BAD_EXAMPLES]
                invalid_format_percents.append(invalid_email_percent)
                issue_score = max(issue_score, invalid_email_percent)
                _append_issue(dataset_issues, "email_format_irregularities")

        if is_phone_candidate:
            invalid_phone_percent = round(float(non_null_count - stats["phone_valid_count"]) / row_count_safe * 100, 2)
            if invalid_phone_percent > 0:
                profile["invalid_phone_percent"] = invalid_phone_percent
                profile["examples"] = stats["invalid_phone_examples"][:MAX_BAD_EXAMPLES]
                invalid_format_percents.append(invalid_phone_percent)
                issue_score = max(issue_score, invalid_phone_percent)
                _append_issue(dataset_issues, "phone_format_irregularities")

            phone_format_inconsistency_percent = round(
                float(stats["phone_format_inconsistency_count"]) / non_null_count * 100,
                2,
            )
            if phone_format_inconsistency_percent > 0:
                profile["phone_format_inconsistency_percent"] = phone_format_inconsistency_percent
                if "examples" not in profile:
                    profile["examples"] = stats["phone_format_examples"][:MAX_BAD_EXAMPLES]
                format_inconsistency_percents.append(phone_format_inconsistency_percent)
                issue_score = max(issue_score, phone_format_inconsistency_percent)
                _append_issue(dataset_issues, "phone_format_normalization_needed")

        if is_date_candidate:
            invalid_date_percent = round(float(non_null_count - stats["date_valid_count"]) / row_count_safe * 100, 2)
            if invalid_date_percent > 0:
                profile["invalid_date_percent"] = invalid_date_percent
                profile["examples"] = stats["invalid_date_examples"][:MAX_BAD_EXAMPLES]
                invalid_format_percents.append(invalid_date_percent)
                issue_score = max(issue_score, invalid_date_percent)
                _append_issue(dataset_issues, "mixed_date_formats")

            date_format_inconsistency_percent = round(
                float(stats["date_format_inconsistency_count"]) / non_null_count * 100,
                2,
            )
            if date_format_inconsistency_percent > 0:
                profile["date_format_inconsistency_percent"] = date_format_inconsistency_percent
                if "examples" not in profile:
                    profile["examples"] = stats["date_format_examples"][:MAX_BAD_EXAMPLES]
                format_inconsistency_percents.append(date_format_inconsistency_percent)
                issue_score = max(issue_score, date_format_inconsistency_percent)
                _append_issue(dataset_issues, "date_format_normalization_needed")

        if is_age_candidate:
            invalid_age_percent = round(float(non_null_count - stats["age_valid_count"]) / row_count_safe * 100, 2)
            if invalid_age_percent > 0:
                profile["invalid_age_percent"] = invalid_age_percent
                profile["invalid_integer_percent"] = invalid_age_percent
                profile["examples"] = stats["invalid_age_examples"][:MAX_BAD_EXAMPLES]
                invalid_format_percents.append(invalid_age_percent)
                numeric_parse_failure_percents.append(invalid_age_percent)
                issue_score = max(issue_score, invalid_age_percent)
                _append_issue(dataset_issues, "age_value_irregularities")

            age_format_inconsistency_percent = round(
                float(stats["age_format_inconsistency_count"]) / non_null_count * 100,
                2,
            )
            if age_format_inconsistency_percent > 0:
                profile["age_format_inconsistency_percent"] = age_format_inconsistency_percent
                if "examples" not in profile:
                    profile["examples"] = stats["age_format_examples"][:MAX_BAD_EXAMPLES]
                format_inconsistency_percents.append(age_format_inconsistency_percent)
                issue_score = max(issue_score, age_format_inconsistency_percent)
                _append_issue(dataset_issues, "age_normalization_needed")

        if expected_type == "integer" and not is_age_candidate:
            invalid_integer_percent = round(
                float(non_null_count - stats["integer_valid_count"]) / row_count_safe * 100,
                2,
            )
            if invalid_integer_percent > 0:
                profile["invalid_integer_percent"] = invalid_integer_percent
                profile["examples"] = stats["invalid_integer_examples"][:MAX_BAD_EXAMPLES]
                invalid_format_percents.append(invalid_integer_percent)
                numeric_parse_failure_percents.append(invalid_integer_percent)
                issue_score = max(issue_score, invalid_integer_percent)
                _append_issue(dataset_issues, "integer_validation_needed")

        if expected_type == "float":
            invalid_float_percent = round(
                float(non_null_count - stats["float_valid_count"]) / row_count_safe * 100,
                2,
            )
            if invalid_float_percent > 0:
                profile["invalid_float_percent"] = invalid_float_percent
                profile["examples"] = stats["invalid_float_examples"][:MAX_BAD_EXAMPLES]
                invalid_format_percents.append(invalid_float_percent)
                numeric_parse_failure_percents.append(invalid_float_percent)
                issue_score = max(issue_score, invalid_float_percent)
                _append_issue(dataset_issues, "float_validation_needed")

        if is_numeric_candidate and not is_phone_candidate and not is_age_candidate and expected_type not in {"integer", "float"}:
            invalid_numeric_percent = round(float(non_null_count - stats["numeric_valid_count"]) / row_count_safe * 100, 2)
            if invalid_numeric_percent > 0:
                profile["invalid_numeric_percent"] = invalid_numeric_percent
                profile["examples"] = stats["invalid_numeric_examples"][:MAX_BAD_EXAMPLES]
                numeric_parse_failure_percents.append(invalid_numeric_percent)
                issue_score = max(issue_score, invalid_numeric_percent)
                _append_issue(dataset_issues, "numeric_parse_failures")

            numeric_format_inconsistency_percent = round(
                float(stats["numeric_format_inconsistency_count"]) / non_null_count * 100,
                2,
            )
            if numeric_format_inconsistency_percent > 0:
                profile["numeric_format_inconsistency_percent"] = numeric_format_inconsistency_percent
                if "examples" not in profile:
                    profile["examples"] = stats["numeric_format_examples"][:MAX_BAD_EXAMPLES]
                format_inconsistency_percents.append(numeric_format_inconsistency_percent)
                issue_score = max(issue_score, numeric_format_inconsistency_percent)
                _append_issue(dataset_issues, "numeric_format_normalization_needed")

        is_semantic_text_candidate = (
            _candidate_by_name(column_name_lower, TEXT_SEMANTIC_COLUMN_HINTS)
            and not any((is_email_candidate, is_phone_candidate, is_date_candidate, is_numeric_candidate))
        )
        if is_semantic_text_candidate:
            semantic_type_mismatch_percent = round(
                float(non_null_count - stats["text_like_count"]) / row_count_safe * 100,
                2,
            )
            if semantic_type_mismatch_percent > 0:
                profile["semantic_type_mismatch_percent"] = semantic_type_mismatch_percent
                profile["examples"] = stats["semantic_type_mismatch_examples"][:MAX_BAD_EXAMPLES]
                invalid_format_percents.append(semantic_type_mismatch_percent)
                issue_score = max(issue_score, semantic_type_mismatch_percent)
                _append_issue(dataset_issues, "header_type_mismatches")

        if _candidate_by_name(column_name_lower, KEY_COLUMN_HINTS):
            key_duplicate_percent = round(float(stats["key_duplicate_count"]) / non_null_count * 100, 2) if non_null_count else 0.0
            if key_duplicate_percent > 0:
                profile["key_duplicate_percent"] = key_duplicate_percent
                issue_score = max(issue_score, key_duplicate_percent)
                _append_issue(dataset_issues, "key_column_duplicates")

        unique_count = len(stats["unique_values"]) if stats["track_uniques"] else 61
        lower_unique_count = len(stats["lower_unique_values"]) if stats["track_uniques"] else 61
        unique_ratio = (unique_count / non_null_count) if non_null_count else 0.0
        average_length = (float(stats["text_length_sum"]) / non_null_count) if non_null_count else 0.0
        is_moderate_cardinality = (
            stats["track_uniques"]
            and unique_count <= 50
            and unique_ratio <= 0.5
            and average_length <= 80
        )
        if is_moderate_cardinality and not any((is_email_candidate, is_phone_candidate, is_date_candidate, is_numeric_candidate)):
            whitespace_inconsistency_percent = round(float(stats["whitespace_inconsistency_count"]) / non_null_count * 100, 2)
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
                if "examples" not in profile:
                    profile["examples"] = stats["text_examples"][:MAX_BAD_EXAMPLES]
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
        "average_format_inconsistency_percent": _mean(format_inconsistency_percents),
        "average_consistency_issue_percent": _mean(text_inconsistency_percents + format_inconsistency_percents),
        "average_numeric_parse_failure_percent": _mean(numeric_parse_failure_percents),
        "columns": compact_columns,
        "dataset_issues": dataset_issues,
        "sample_rows": sample_rows[:MAX_SAMPLE_ROWS],
    }


def build_dataset_profile(df: pd.DataFrame) -> dict[str, Any]:
    return build_dataset_profile_from_chunks([df])


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
    consistency_issue_percent = float(
        profile.get(
            "average_consistency_issue_percent",
            profile.get("average_text_inconsistency_percent", 0.0),
        )
    )

    penalty = 0.0
    penalty += min(20.0, 20.0 * (float(profile.get("average_null_percent", 0.0)) / 100.0))
    penalty += min(20.0, 20.0 * (float(profile.get("average_invalid_format_percent", 0.0)) / 100.0))
    penalty += min(10.0, 10.0 * (float(profile.get("duplicate_row_percent", 0.0)) / 100.0))
    penalty += min(10.0, 10.0 * (consistency_issue_percent / 100.0))
    penalty += min(10.0, 10.0 * (float(profile.get("average_numeric_parse_failure_percent", 0.0)) / 100.0))
    penalty += min(10.0, 10.0 * (affected_column_percent / 100.0))
    penalty += min(15.0, 15.0 * (max_column_issue_percent / 100.0))
    penalty += min(5.0, 5.0 * (dataset_issue_count / 6.0))

    return max(0, min(100, int(round(100.0 - penalty))))


def _columns_with_metric(
    columns: list[dict[str, Any]],
    metric_name: str,
    *,
    threshold: float = 0.0,
) -> list[str]:
    return [
        str(column.get("name"))
        for column in columns
        if str(column.get("name", "")).strip()
        and float(column.get(metric_name, 0.0)) > threshold
    ]


def _format_column_targets(column_names: list[str], *, limit: int = 4) -> str:
    unique_names = list(dict.fromkeys(column_names))
    if not unique_names:
        return "the affected columns"

    shown_names = [f"'{name}'" for name in unique_names[:limit]]
    if len(unique_names) > limit:
        shown_names.append(f"and {len(unique_names) - limit} more column(s)")

    if len(shown_names) == 1:
        return shown_names[0]

    return ", ".join(shown_names)


def _group_column_targets(column_names: list[str], *, group_size: int = 4) -> list[str]:
    unique_names = list(dict.fromkeys(column_names))
    if not unique_names:
        return []

    return [
        _format_column_targets(unique_names[index:index + group_size], limit=group_size)
        for index in range(0, len(unique_names), group_size)
    ]


def generate_rule_based_suggestions(
    profile: dict[str, Any],
    *,
    max_suggestions: int = 10,
) -> list[DataSuggestion]:
    suggestions: list[DataSuggestion] = []
    columns = profile.get("columns", [])
    duplicate_row_percent = float(profile.get("duplicate_row_percent", 0.0))

    boolean_issue_columns = _columns_with_metric(columns, "invalid_boolean_percent")
    for boolean_targets in _group_column_targets(boolean_issue_columns):
        suggestions.append(
            DataSuggestion(
                issue_description=(
                    f"Boolean-like column(s) {boolean_targets} contain values outside the allowed set True, False, 1, or 0."
                ),
                priority="High",
                resolution_prompt=(
                    f"Only in column(s) {boolean_targets}, accept only True, False, 1, and 0 case-insensitively, "
                    f"replace every other value with '{settings.NULL_OUTPUT_TOKEN}', and leave unrelated columns untouched."
                ),
            )
        )

    phone_issue_columns = list(
        dict.fromkeys(
            _columns_with_metric(columns, "invalid_phone_percent", threshold=5.0)
            + _columns_with_metric(columns, "phone_format_inconsistency_percent", threshold=5.0)
        )
    )
    for phone_targets in _group_column_targets(phone_issue_columns):
        suggestions.append(
            DataSuggestion(
                issue_description=(
                    f"Phone-like column(s) {phone_targets} contain invalid values or inconsistent formatting."
                ),
                priority="High",
                resolution_prompt=(
                    f"Only in column(s) {phone_targets}, standardize phone values into one consistent phone format, "
                    "remove punctuation noise, preserve country codes when present, keep valid phone-like values, "
                    "and leave unrelated columns untouched."
                ),
            )
        )

    date_issue_columns = list(
        dict.fromkeys(
            _columns_with_metric(columns, "invalid_date_percent")
            + _columns_with_metric(columns, "date_format_inconsistency_percent")
        )
    )
    for date_targets in _group_column_targets(date_issue_columns):
        suggestions.append(
            DataSuggestion(
                issue_description=(
                    f"Date-like column(s) {date_targets} contain invalid values or inconsistent date formatting."
                ),
                priority="High",
                resolution_prompt=(
                    f"Only in column(s) {date_targets}, parse values with pandas.to_datetime(errors='coerce'), "
                    f"keep parseable dates, replace invalid dates with '{settings.NULL_OUTPUT_TOKEN}', "
                    f"and format valid dates to {settings.DATE_OUTPUT_FORMAT} without modifying non-date columns."
                ),
            )
        )

    missing_value_columns = _columns_with_metric(columns, "null_percent", threshold=10.0)
    for missing_targets in _group_column_targets(missing_value_columns):
        suggestions.append(
            DataSuggestion(
                issue_description=(
                    f"Column(s) {missing_targets} contain a meaningful share of missing, blank, or placeholder values."
                ),
                priority="Medium",
                resolution_prompt=(
                    f"Only in column(s) {missing_targets}, normalize blank, null-like, and placeholder values to "
                    f"the single token '{settings.NULL_OUTPUT_TOKEN}', replace only genuinely missing entries, "
                    "and do not invent or infer missing data."
                ),
            )
        )

    integer_issue_columns = [
        str(column.get("name"))
        for column in columns
        if float(column.get("invalid_integer_percent", 0.0)) > 0.0
        and float(column.get("invalid_age_percent", 0.0)) == 0.0
    ]
    for integer_targets in _group_column_targets(integer_issue_columns):
        suggestions.append(
            DataSuggestion(
                issue_description=(
                    f"Integer-like column(s) {integer_targets} contain non-integer values under strict validation."
                ),
                priority="High",
                resolution_prompt=(
                    f"Only in column(s) {integer_targets}, keep only valid integers in strict mode, "
                    f"do not auto-convert string values such as '123', replace non-integer values with '{settings.NULL_OUTPUT_TOKEN}', "
                    "and leave unrelated columns untouched."
                ),
            )
        )

    float_issue_columns = _columns_with_metric(columns, "invalid_float_percent")
    for float_targets in _group_column_targets(float_issue_columns):
        suggestions.append(
            DataSuggestion(
                issue_description=(
                    f"Float-like column(s) {float_targets} contain values that are not valid floats."
                ),
                priority="High",
                resolution_prompt=(
                    f"Only in column(s) {float_targets}, keep only valid float values, replace every other value with '{settings.NULL_OUTPUT_TOKEN}', "
                    "and leave unrelated columns untouched."
                ),
            )
        )

    age_issue_columns = list(
        dict.fromkeys(
            _columns_with_metric(columns, "invalid_age_percent")
            + _columns_with_metric(columns, "age_format_inconsistency_percent")
        )
    )
    for age_targets in _group_column_targets(age_issue_columns):
        suggestions.append(
            DataSuggestion(
                issue_description=(
                    f"Age-like column(s) {age_targets} contain non-integer, negative, impossible, or inconsistent age values."
                ),
                priority="High",
                resolution_prompt=(
                    f"Only in column(s) {age_targets}, keep only strict integer ages between {AGE_MIN_VALUE} and {AGE_MAX_VALUE}, "
                    f"do not auto-convert string values such as '20' or 'twenty', replace negative, impossible, or non-integer age values "
                    f"with '{settings.NULL_OUTPUT_TOKEN}', and leave unrelated columns untouched."
                ),
            )
        )

    semantic_type_columns = _columns_with_metric(columns, "semantic_type_mismatch_percent")
    for semantic_targets in _group_column_targets(semantic_type_columns):
        suggestions.append(
            DataSuggestion(
                issue_description=(
                    f"Text-like column(s) {semantic_targets} contain values that do not match the header's expected text/string type."
                ),
                priority="High",
                resolution_prompt=(
                    f"Only in column(s) {semantic_targets}, use the column headers as the type guide, keep valid text-like values as trimmed strings, "
                    f"and replace values that are only numbers or obvious non-text placeholders with '{settings.NULL_OUTPUT_TOKEN}'. "
                    "Do not modify unrelated columns."
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
        text_columns = list(
            dict.fromkeys(
                _columns_with_metric(columns, "casing_inconsistency_percent")
                + _columns_with_metric(columns, "whitespace_inconsistency_percent")
            )
        )
        for text_targets in _group_column_targets(text_columns):
            suggestions.append(
                DataSuggestion(
                    issue_description=(
                        f"Moderate-cardinality text column(s) {text_targets} show casing or whitespace inconsistencies."
                    ),
                    priority="Medium",
                    resolution_prompt=(
                        f"Only in column(s) {text_targets}, trim leading and trailing whitespace and standardize repeated "
                        "text values that clearly represent the same meaning to one canonical casing. Preserve free-text "
                        "content and do not rewrite unrelated long-form text."
                    ),
                )
            )

    numeric_issue_columns = list(
        dict.fromkeys(
            _columns_with_metric(columns, "invalid_numeric_percent", threshold=5.0)
            + _columns_with_metric(columns, "numeric_format_inconsistency_percent", threshold=5.0)
        )
    )
    for numeric_targets in _group_column_targets(numeric_issue_columns):
        suggestions.append(
            DataSuggestion(
                issue_description=(
                    f"Numeric-like column(s) {numeric_targets} contain formatting noise or inconsistent numeric presentation."
                ),
                priority="Medium",
                resolution_prompt=(
                    f"Only in column(s) {numeric_targets}, remove formatting noise such as currency symbols, spaces, "
                    "commas, and separators when they are presentation-only, preserve valid numeric values, "
                    "and leave unrelated non-numeric columns untouched."
                ),
            )
        )

    return suggestions[:max(1, max_suggestions)]
