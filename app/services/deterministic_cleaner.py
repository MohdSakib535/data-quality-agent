import re
from typing import List

import pandas as pd

from app.schemas.models import DataSuggestion, DatasetAnalysisPayload


def _to_snake_case(value: str) -> str:
    normalized = re.sub(r"[^0-9a-zA-Z]+", "_", value.strip())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized.lower()


def _priority_from_ratio(ratio: float, medium_threshold: float, high_threshold: float) -> str:
    if ratio >= high_threshold:
        return "High"
    if ratio >= medium_threshold:
        return "Medium"
    return "Low"


def _build_suggestion(issue_description: str, priority: str, resolution_prompt: str) -> DataSuggestion:
    return DataSuggestion(
        issue_description=issue_description,
        priority=priority,
        resolution_prompt=resolution_prompt,
    )


def analyze_dataset_deterministically(df: pd.DataFrame) -> DatasetAnalysisPayload:
    suggestions: List[DataSuggestion] = []
    penalties = 0.0

    row_count = len(df)
    column_count = len(df.columns)
    total_cells = max(row_count * max(column_count, 1), 1)

    missing_ratio = float(df.isna().sum().sum()) / total_cells
    if missing_ratio > 0:
        priority = _priority_from_ratio(missing_ratio, 0.05, 0.15)
        suggestions.append(
            _build_suggestion(
                issue_description=(
                    f"The dataset has {missing_ratio:.1%} missing values across {row_count} rows and "
                    f"{column_count} columns."
                ),
                priority=priority,
                resolution_prompt=(
                    "Standardize missing value tokens, convert blank placeholders to null, and fill or remove "
                    "missing values only where appropriate for each column type."
                ),
            )
        )
        penalties += min(35.0, missing_ratio * 100)

    if row_count > 0:
        duplicate_ratio = float(df.duplicated().sum()) / row_count
        if duplicate_ratio > 0:
            priority = _priority_from_ratio(duplicate_ratio, 0.02, 0.1)
            suggestions.append(
                _build_suggestion(
                    issue_description=f"The dataset contains {duplicate_ratio:.1%} fully duplicated rows.",
                    priority=priority,
                    resolution_prompt=(
                        "Identify exact duplicate rows and remove redundant copies while keeping one canonical record."
                    ),
                )
            )
            penalties += min(20.0, duplicate_ratio * 120)

    header_issues = [column for column in df.columns if _to_snake_case(str(column)) != str(column).strip()]
    if header_issues:
        suggestions.append(
            _build_suggestion(
                issue_description=(
                    "Some column names are not normalized and may contain spaces, punctuation, or inconsistent casing."
                ),
                priority="Medium",
                resolution_prompt=(
                    "Normalize all column headers to lowercase snake_case and trim surrounding whitespace."
                ),
            )
        )
        penalties += min(8.0, float(len(header_issues)) * 2.0)

    object_columns = df.select_dtypes(include=["object", "string"]).columns.tolist()
    whitespace_columns = []
    inconsistent_case_columns = []

    for column in object_columns:
        series = df[column].dropna().astype(str)
        if series.empty:
            continue

        if series.str.match(r"^\s|\s$", na=False).any():
            whitespace_columns.append(column)

        normalized_lower = series.str.strip().str.lower()
        unique_original = set(series.tolist())
        unique_normalized = set(normalized_lower.tolist())
        if unique_normalized and len(unique_original) > len(unique_normalized):
            inconsistent_case_columns.append(column)

    if whitespace_columns:
        suggestions.append(
            _build_suggestion(
                issue_description=(
                    "Some text columns contain leading or trailing whitespace that can break matching and grouping."
                ),
                priority="Medium",
                resolution_prompt=(
                    f"Trim leading and trailing whitespace in these columns: {', '.join(whitespace_columns[:6])}."
                ),
            )
        )
        penalties += min(10.0, float(len(whitespace_columns)) * 2.0)

    if inconsistent_case_columns:
        suggestions.append(
            _build_suggestion(
                issue_description=(
                    "Some text columns appear to use inconsistent capitalization for equivalent values."
                ),
                priority="Low" if len(inconsistent_case_columns) == 1 else "Medium",
                resolution_prompt=(
                    f"Standardize capitalization and spelling for repeated text values in these columns: "
                    f"{', '.join(inconsistent_case_columns[:6])}."
                ),
            )
        )
        penalties += min(12.0, float(len(inconsistent_case_columns)) * 2.0)

    quality_score = max(0, min(100, int(round(100 - penalties))))
    return DatasetAnalysisPayload(quality_score=quality_score, suggestions=suggestions[:5])
