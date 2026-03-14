import json
import re
from typing import List, Dict, Any
import pandas as pd
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from app.prompts.cleaning_prompts import analysis_prompt_template, clean_data_prompt_template
from app.schemas.models import DataSuggestion, DatasetAnalysisPayload, DatasetAnalysisResponse
from app.core.config import settings
from app.services.deterministic_cleaner import (
    analyze_dataset_deterministically,
)

try:
    from langchain_ollama import ChatOllama
except ImportError:
    from langchain_community.chat_models import ChatOllama

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


def _merge_analysis_payloads(
    deterministic: DatasetAnalysisPayload,
    ai_payload: DatasetAnalysisPayload | None,
) -> DatasetAnalysisPayload:
    if ai_payload is None:
        return deterministic

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
    Iterate through the dataframe in small batches, applying the user's cleaning prompt.
    """
    llm = ChatOllama(model=settings.OLLAMA_MODEL, base_url=settings.OLLAMA_BASE_URL, temperature=0.0)
    chain = clean_data_prompt_template | llm | StrOutputParser()
    
    batch_size = 20
    cleaned_rows = []
    
    # Fill NAs to None for JSON
    df_clean = df.where(pd.notnull(df), None)
    records = df_clean.to_dict(orient="records")
    
    for i in range(0, len(records), batch_size):
        batch = records[i:i+batch_size]
        batch_json = json.dumps(batch, indent=2)
        
        try:
            cleaned_batch = _invoke_cleaning_batch(
                chain=chain,
                batch_json=batch_json,
                user_prompt=user_prompt,
                expected_rows=len(batch),
            )
            cleaned_rows.extend(cleaned_batch)
        except Exception as e:
            print(f"Error cleaning batch {i} to {i+batch_size}: {e}")
            cleaned_rows.extend(batch)
            
    return pd.DataFrame(cleaned_rows)
