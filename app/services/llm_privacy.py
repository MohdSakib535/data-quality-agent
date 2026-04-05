import re
from typing import Any

import pandas as pd

EMAIL_PATTERN = re.compile(r"^[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}$", re.IGNORECASE)
PHONE_DIGITS_PATTERN = re.compile(r"^\d{10,15}$")
DATE_LIKE_PATTERN = re.compile(r"^\d{1,4}[-/]\d{1,2}[-/]\d{1,4}(?:[ T]\d{1,2}:\d{2}(?::\d{2})?)?$")
ID_LIKE_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9\-_]{7,}$", re.IGNORECASE)

SENSITIVE_COLUMN_HINTS = (
    "name",
    "full_name",
    "first_name",
    "last_name",
    "email",
    "phone",
    "mobile",
    "contact",
    "address",
    "street",
    "zip",
    "postal",
    "dob",
    "birth",
    "aadhaar",
    "aadhar",
    "pan",
    "passport",
    "ssn",
    "account",
    "card",
    "cvv",
    "iban",
    "swift",
    "ifsc",
    "password",
    "secret",
    "token",
    "apikey",
    "api_key",
)


def is_sensitive_column_name(column_name: str) -> bool:
    normalized = str(column_name).strip().lower()
    return any(hint in normalized for hint in SENSITIVE_COLUMN_HINTS)


def _is_empty(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "") or pd.isna(value)


def _looks_like_phone(value: str) -> bool:
    digits = re.sub(r"\D+", "", value)
    return bool(PHONE_DIGITS_PATTERN.fullmatch(digits))


def _looks_like_numeric_identifier(value: str) -> bool:
    digits = re.sub(r"\D+", "", value)
    return len(digits) >= 8 and len(digits) == len(value.strip())


def value_looks_sensitive(value: Any) -> bool:
    if _is_empty(value):
        return False

    if isinstance(value, (int, float)):
        return False

    text = str(value).strip()
    if not text:
        return False

    if EMAIL_PATTERN.fullmatch(text):
        return True
    if _looks_like_phone(text):
        return True
    if _looks_like_numeric_identifier(text):
        return True
    if ID_LIKE_PATTERN.fullmatch(text):
        return True
    return False


def redact_value_for_analysis(value: Any) -> Any:
    if _is_empty(value):
        return None

    if isinstance(value, bool):
        return "<BOOL>"
    if isinstance(value, int):
        return "<INT>"
    if isinstance(value, float):
        return "<FLOAT>"

    text = str(value).strip()
    if EMAIL_PATTERN.fullmatch(text):
        return "<EMAIL>"
    if _looks_like_phone(text):
        return "<PHONE>"
    if DATE_LIKE_PATTERN.fullmatch(text):
        return "<DATE>"
    if _looks_like_numeric_identifier(text) or ID_LIKE_PATTERN.fullmatch(text):
        return "<ID>"

    word_count = len([part for part in re.split(r"\s+", text) if part])
    return f"<TEXT len={len(text)} words={word_count}>"


def redact_analysis_profile(profile: dict[str, Any]) -> dict[str, Any]:
    redacted_columns = []
    for column in profile.get("columns", []):
        redacted_column = dict(column)
        if "examples" in redacted_column:
            redacted_column["examples"] = [
                redact_value_for_analysis(example)
                for example in redacted_column.get("examples", [])
            ]
        redacted_columns.append(redacted_column)

    redacted_sample_rows = []
    for row in profile.get("sample_rows", [])[:5]:
        if not isinstance(row, dict):
            continue
        redacted_sample_rows.append(
            {
                key: redact_value_for_analysis(value)
                for key, value in row.items()
            }
        )

    return {
        **profile,
        "columns": redacted_columns,
        "sample_rows": redacted_sample_rows,
    }


def select_llm_safe_columns(df: pd.DataFrame, ai_columns: list[str]) -> list[str]:
    safe_columns: list[str] = []
    for column in ai_columns:
        if is_sensitive_column_name(column):
            continue

        series = df[column] if column in df.columns else None
        if series is None:
            continue

        sample_values = [value for value in series.head(25).tolist() if not _is_empty(value)]
        if any(value_looks_sensitive(value) for value in sample_values):
            continue

        safe_columns.append(column)

    return safe_columns
