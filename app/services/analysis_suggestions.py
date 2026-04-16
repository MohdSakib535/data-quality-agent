import re

from app.models.analysis_suggestion import AnalysisSuggestion
from app.schemas.job import DataSuggestion, SuggestionDetailResponse


def coalesce_clean_quality_score(
    current_score: int,
    *,
    raw_score: int | None,
    previous_clean_score: int | None,
    changes_detected: bool = False,
) -> int:
    """
    Keep cleaned-data quality scoring monotonic across the analysis flow.

    Clean analysis should represent progress over the original upload and should
    not regress across repeated clean/analyze cycles due to heuristic scoring
    shifts or suggestion suppression. When a clean run materially changed the
    dataset but the heuristic score stays flat, nudge the score upward by one
    point so the API reflects visible cleaning progress.
    """
    baseline_score = max(
        current_score,
        raw_score if raw_score is not None else current_score,
        previous_clean_score if previous_clean_score is not None else current_score,
    )
    if changes_detected and baseline_score == max(
        raw_score if raw_score is not None else baseline_score,
        previous_clean_score if previous_clean_score is not None else baseline_score,
    ):
        return min(100, baseline_score + 1)

    return baseline_score


def _normalize_prompt_text(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value.strip().lower())


def classify_issue_category(value: str | None) -> str | None:
    normalized = _normalize_prompt_text(value)
    if not normalized:
        return None

    category_terms = {
        "age": ("age", "ages", "age-like", "age values", "whole-number ages", "plausible ages", "spelled-out age"),
        "boolean": ("boolean", "true", "false", "1, and 0", "1 or 0"),
        "integer": ("strict integer", "valid integers", "non-integer", "integer-like"),
        "float": ("valid float", "float-like", "not valid floats"),
        "phone": ("phone", "mobile", "contact number", "country code", "phone-like"),
        "date": ("date", "dates", "datetime", "timestamp", "date format"),
        "missing": ("missing", "null", "blank", "empty", "placeholder"),
        "duplicate": ("duplicate", "duplicates", "deduplicate", "redundant copies"),
        "text": ("whitespace", "casing", "text normalization", "normalize text", "repeated text"),
        "numeric": ("numeric", "numbers", "currency", "separators", "parse cleanly as numbers"),
        "email": ("email", "e-mail", "mail address"),
        "schema_type": (
            "header semantics",
            "type guide",
            "text-like values",
            "text/string type",
            "column headers as the type guide",
            "non-text placeholders",
        ),
    }

    for category, terms in category_terms.items():
        if any(term in normalized for term in terms):
            return category
    return None


def should_suppress_suggestion(last_clean_prompt: str | None, suggestion: DataSuggestion) -> bool:
    normalized_last_prompt = _normalize_prompt_text(last_clean_prompt)
    if not normalized_last_prompt:
        return False

    if _normalize_prompt_text(suggestion.resolution_prompt) == normalized_last_prompt:
        return True

    last_category = classify_issue_category(last_clean_prompt)
    suggestion_category = (
        classify_issue_category(suggestion.resolution_prompt)
        or classify_issue_category(suggestion.issue_description)
    )
    return bool(last_category and suggestion_category and last_category == suggestion_category)


def _extract_target_columns_from_text(value: str | None) -> list[str]:
    if not value:
        return []

    patterns = [
        r"column\(s\)\s+(.*?),(?:\s|$)",
        r"column\s+(.+?),(?:\s|$)",
    ]

    extracted_targets: list[str] = []
    for pattern in patterns:
        matches = re.findall(pattern, value, flags=re.IGNORECASE)
        for match in matches:
            extracted_targets.extend(re.findall(r"'([^']+)'", match))

    return list(dict.fromkeys(target.strip() for target in extracted_targets if target.strip()))


def extract_target_columns_from_suggestion(suggestion: DataSuggestion) -> list[str]:
    prompt_targets = _extract_target_columns_from_text(suggestion.resolution_prompt)
    if prompt_targets:
        return prompt_targets

    return _extract_target_columns_from_text(suggestion.issue_description)


def classify_cleaning_prompt_type(suggestion: DataSuggestion) -> str | None:
    category = (
        classify_issue_category(suggestion.resolution_prompt)
        or classify_issue_category(suggestion.issue_description)
    )
    category_to_type = {
        "age": "age_normalization",
        "boolean": "boolean_validation",
        "integer": "integer_validation",
        "float": "float_validation",
        "phone": "phone_normalization",
        "date": "date_normalization",
        "missing": "missing_value_normalization",
        "duplicate": "duplicate_removal",
        "text": "text_normalization",
        "numeric": "numeric_normalization",
        "email": "email_normalization",
        "schema_type": "header_type_normalization",
    }
    return category_to_type.get(category, "generic_cleaning" if suggestion.resolution_prompt else None)


def build_data_suggestion(
    suggestion: DataSuggestion,
    *,
    suggestion_id: str | None,
) -> DataSuggestion:
    return DataSuggestion(
        id=suggestion_id,
        issue_description=suggestion.issue_description,
        priority=suggestion.priority,
        resolution_prompt=suggestion.resolution_prompt,
        cleaning_prompt_type=classify_cleaning_prompt_type(suggestion),
        target_columns=extract_target_columns_from_suggestion(suggestion),
    )


def build_suggestion_detail_response(suggestion: AnalysisSuggestion) -> SuggestionDetailResponse:
    base_suggestion = DataSuggestion(
        id=suggestion.id,
        issue_description=suggestion.issue_description,
        priority=suggestion.priority,
        resolution_prompt=suggestion.resolution_prompt,
    )
    return SuggestionDetailResponse(
        id=suggestion.id,
        job_id=suggestion.job_id,
        source_type=suggestion.source_type,
        issue_description=suggestion.issue_description,
        priority=suggestion.priority,
        resolution_prompt=suggestion.resolution_prompt,
        cleaning_prompt_type=classify_cleaning_prompt_type(base_suggestion),
        target_columns=extract_target_columns_from_suggestion(base_suggestion),
        created_at=suggestion.created_at,
        updated_at=suggestion.updated_at,
    )
