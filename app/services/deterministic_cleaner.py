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
                    "Review the dataset for missing or placeholder values across all affected columns and replace "
                    "them with the literal string N/A wherever the value is missing. Treat blank strings, "
                    "whitespace-only cells, null, NULL, NA, -, unknown, and other visibly empty placeholders as "
                    "missing values. Apply the replacement consistently across the dataset, preserve all non-empty "
                    "valid values as they are, and do not invent or guess any new data beyond converting missing "
                    "entries to N/A."
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
                        "Identify exact duplicate records across the dataset and remove redundant copies while "
                        "keeping one canonical version of each repeated row. Preserve legitimate repeated values "
                        "that are not true duplicates, and avoid merging rows unless all corresponding fields "
                        "represent the same record."
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
                    "Normalize all column headers consistently across the dataset. Trim surrounding whitespace, "
                    "remove punctuation noise, convert names to lowercase snake_case, and ensure each header is "
                    "clear, stable, and unique without changing the meaning of the column."
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
                    "Trim leading and trailing whitespace in all affected text fields across the dataset. "
                    "Preserve meaningful internal spacing unless it is clearly accidental, and ensure values that "
                    "become empty after trimming are handled consistently as blank or missing data."
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
                    "Standardize capitalization, spelling, and formatting for repeated text values that represent "
                    "the same meaning. Apply one canonical form consistently across the dataset while preserving "
                    "genuinely distinct values and avoiding unsupported corrections."
                ),
            )
        )
        penalties += min(12.0, float(len(inconsistent_case_columns)) * 2.0)

    quality_score = max(0, min(100, int(round(100 - penalties))))
    return DatasetAnalysisPayload(quality_score=quality_score, suggestions=suggestions[:5])
