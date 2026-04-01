import re
from datetime import datetime
from typing import Any

from sqlalchemy import delete, select

from app.db.session import AsyncSessionLocal
from app.models.semantic_column_metadata import SemanticColumnMetadata

_STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}

_EMAIL_PATTERN = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
_UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    output: list[str] = []
    for value in values:
        normalized = value.strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(value.strip())
    return output


def _normalize_column_tokens(column_name: str) -> list[str]:
    normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", column_name)
    normalized = re.sub(r"[^a-zA-Z0-9]+", " ", normalized).strip().lower()
    return [token for token in normalized.split() if token]


def _tokenize(text: str) -> set[str]:
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return {token for token in tokens if len(token) > 1 and token not in _STOPWORDS}


def _singularize(token: str) -> str:
    if token.endswith("ies") and len(token) > 3:
        return token[:-3] + "y"
    if token.endswith("s") and len(token) > 3 and not token.endswith("ss"):
        return token[:-1]
    return token


def _pluralize(token: str) -> str:
    if token.endswith("y") and len(token) > 2 and token[-2] not in "aeiou":
        return token[:-1] + "ies"
    if token.endswith("s"):
        return token
    return token + "s"


def _matches_date(value: str) -> bool:
    value = value.strip()
    date_formats = [
        "%Y-%m-%d",
        "%d-%m-%Y",
        "%m-%d-%Y",
        "%Y/%m/%d",
        "%d/%m/%Y",
        "%m/%d/%Y",
    ]
    for pattern in date_formats:
        try:
            datetime.strptime(value, pattern)
            return True
        except ValueError:
            continue
    return False


def _matches_datetime(value: str) -> bool:
    candidate = value.strip().replace("Z", "+00:00")
    try:
        datetime.fromisoformat(candidate)
        return "T" in candidate or ":" in candidate
    except ValueError:
        return False


def _matches_phone(value: str) -> bool:
    digits = re.sub(r"\D", "", value)
    return 10 <= len(digits) <= 15


def _matches_percentage(value: str) -> bool:
    trimmed = value.strip()
    if trimmed.endswith("%"):
        return True
    try:
        numeric = float(trimmed)
        return 0 <= numeric <= 100
    except ValueError:
        return False


def _matches_currency(value: str) -> bool:
    trimmed = value.strip()
    if any(symbol in trimmed for symbol in ("$", "€", "£", "₹", "¥")):
        return True
    try:
        float(trimmed.replace(",", ""))
        return False
    except ValueError:
        return False


def _match_ratio(sample_values: list[str], matcher) -> float:
    if not sample_values:
        return 0.0
    matches = 0
    for value in sample_values:
        if matcher(value):
            matches += 1
    return matches / len(sample_values)


def _header_aliases(tokens: list[str]) -> list[str]:
    if not tokens:
        return []

    aliases: list[str] = []
    phrase = " ".join(tokens)
    aliases.extend(
        [
            phrase,
            "_".join(tokens),
            "-".join(tokens),
            "".join(tokens),
        ]
    )

    for token in tokens:
        aliases.extend([token, _singularize(token), _pluralize(token)])

    if len(tokens) > 1:
        singular_tokens = list(tokens)
        singular_tokens[-1] = _singularize(singular_tokens[-1])
        aliases.append(" ".join(singular_tokens))

        plural_tokens = list(tokens)
        plural_tokens[-1] = _pluralize(plural_tokens[-1])
        aliases.append(" ".join(plural_tokens))

        acronym = "".join(token[0] for token in tokens if token)
        if len(acronym) > 1:
            aliases.append(acronym)

    return aliases


def _semantic_aliases_from_samples(
    tokens: list[str],
    data_type: str,
    sample_values: list[str],
) -> list[str]:
    aliases: list[str] = []

    email_ratio = _match_ratio(sample_values, _EMAIL_PATTERN.match)
    phone_ratio = _match_ratio(sample_values, _matches_phone)
    date_ratio = _match_ratio(sample_values, _matches_date)
    datetime_ratio = _match_ratio(sample_values, _matches_datetime)
    percentage_ratio = _match_ratio(sample_values, _matches_percentage)
    currency_ratio = _match_ratio(sample_values, _matches_currency)
    uuid_ratio = _match_ratio(sample_values, _UUID_PATTERN.match)

    if email_ratio >= 0.6:
        aliases.extend(["email", "email address", "contact email"])
    if phone_ratio >= 0.6:
        aliases.extend(["phone", "mobile", "contact number"])
    if datetime_ratio >= 0.6:
        aliases.extend(["timestamp", "datetime", "date time"])
    elif date_ratio >= 0.6:
        aliases.extend(["date", "day"])
    if percentage_ratio >= 0.6:
        aliases.extend(["percentage", "percent", "rate"])
    if currency_ratio >= 0.6:
        aliases.extend(["amount", "value", "total"])

    if "bool" in data_type.lower() or _match_ratio(sample_values, lambda value: value.lower() in {"true", "false", "yes", "no", "0", "1"}) >= 0.6:
        aliases.extend(["status", "flag", "boolean"])

    token_set = set(tokens)
    if "id" in token_set or uuid_ratio >= 0.6:
        aliases.extend(["identifier", "record id", "reference"])

    if "name" in token_set:
        aliases.extend(["name", "title", "label"])

    if "amount" in token_set or "total" in token_set:
        aliases.extend(["amount", "total", "value"])

    return aliases


def _build_synonyms(
    column_name: str,
    data_type: str = "text",
    sample_values: list[str] | None = None,
) -> list[str]:
    tokens = _normalize_column_tokens(column_name)
    if not tokens:
        return []

    sample_values = sample_values or []
    synonyms: list[str] = []
    synonyms.extend(_header_aliases(tokens))
    synonyms.extend(_semantic_aliases_from_samples(tokens, data_type, sample_values))
    return _dedupe_preserve_order(synonyms)


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return ""
    text_value = str(value).strip()
    if len(text_value) > 80:
        return text_value[:77] + "..."
    return text_value


def _sample_values_for_column(
    row_samples: list[dict[str, Any]],
    column_name: str,
    limit: int = 8,
) -> list[str]:
    seen = set()
    sample_values: list[str] = []

    for row in row_samples:
        if not isinstance(row, dict):
            continue

        value = row.get(column_name)
        rendered = _to_text(value)
        if not rendered:
            continue

        key = rendered.lower()
        if key in seen:
            continue

        seen.add(key)
        sample_values.append(rendered)
        if len(sample_values) >= limit:
            break

    return sample_values


def _build_description(column_name: str, data_type: str, sample_values: list[str]) -> str:
    prefix = f"Column {column_name} stores {data_type} values."
    if not sample_values:
        return prefix
    sample_preview = ", ".join(sample_values[:3])
    return f"{prefix} Example values: {sample_preview}."


def _build_metadata_rows(
    job_id: str,
    table_name: str,
    table_schema: list[dict[str, str]],
    row_schema: list[dict[str, str]] | None,
    row_samples: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    metadata_rows: list[dict[str, Any]] = []
    row_schema = row_schema or []
    row_samples = row_samples or []

    for column in table_schema:
        column_name = str(column.get("column_name", "")).strip()
        data_type = str(column.get("data_type", "text")).strip() or "text"
        if not column_name:
            continue

        metadata_rows.append(
            {
                "job_id": job_id,
                "table_name": table_name,
                "source_type": "table",
                "column_name": column_name,
                "data_type": data_type,
                "synonyms": _build_synonyms(column_name, data_type=data_type, sample_values=[]),
                "sample_values": [],
            }
        )

    for column in row_schema:
        column_name = str(column.get("column_name", "")).strip()
        data_type = str(column.get("data_type", "text")).strip() or "text"
        if not column_name:
            continue

        sample_values = _sample_values_for_column(row_samples, column_name)
        metadata_rows.append(
            {
                "job_id": job_id,
                "table_name": table_name,
                "source_type": "json_row",
                "column_name": column_name,
                "data_type": data_type,
                "synonyms": _build_synonyms(
                    column_name,
                    data_type=data_type,
                    sample_values=sample_values,
                ),
                "sample_values": sample_values,
            }
        )

    for row in metadata_rows:
        row["description"] = _build_description(
            row["column_name"],
            row["data_type"],
            row["sample_values"],
        )

    return metadata_rows


async def ensure_semantic_metadata(
    job_id: str,
    table_name: str,
    table_schema: list[dict[str, str]],
    row_schema: list[dict[str, str]] | None = None,
    row_samples: list[dict[str, Any]] | None = None,
) -> None:
    rows = _build_metadata_rows(
        job_id=job_id,
        table_name=table_name,
        table_schema=table_schema,
        row_schema=row_schema,
        row_samples=row_samples,
    )
    if not rows:
        return

    async with AsyncSessionLocal() as session:
        await session.execute(
            delete(SemanticColumnMetadata).where(
                SemanticColumnMetadata.job_id == job_id,
                SemanticColumnMetadata.table_name == table_name,
            )
        )
        session.add_all(
            [
                SemanticColumnMetadata(
                    job_id=row["job_id"],
                    table_name=row["table_name"],
                    source_type=row["source_type"],
                    column_name=row["column_name"],
                    data_type=row["data_type"],
                    description=row["description"],
                    synonyms=row["synonyms"],
                    sample_values=row["sample_values"],
                )
                for row in rows
            ]
        )
        await session.commit()


async def semantic_metadata_exists(job_id: str, table_name: str) -> bool:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(SemanticColumnMetadata.id).where(
                SemanticColumnMetadata.job_id == job_id,
                SemanticColumnMetadata.table_name == table_name,
            ).limit(1)
        )
        return result.scalar_one_or_none() is not None


def _score_metadata_candidate(
    question: str,
    question_tokens: set[str],
    metadata: SemanticColumnMetadata,
) -> float:
    question_lower = question.lower()
    score = 0.0

    column_name = metadata.column_name or ""
    column_tokens = _tokenize(column_name)
    score += 2.5 * len(column_tokens & question_tokens)

    column_phrase = " ".join(_normalize_column_tokens(column_name))
    if column_phrase and column_phrase in question_lower:
        score += 4.0
    if column_name.lower() in question_lower:
        score += 5.0

    synonyms = metadata.synonyms or []
    for synonym in synonyms:
        synonym_lower = synonym.lower().strip()
        if not synonym_lower:
            continue
        synonym_tokens = _tokenize(synonym_lower)
        overlap = len(synonym_tokens & question_tokens)
        if synonym_lower in question_lower:
            score += 3.5
        if overlap:
            score += 1.6 * overlap / max(len(synonym_tokens), 1)

    sample_values = metadata.sample_values or []
    for sample in sample_values:
        sample_lower = sample.lower().strip()
        if not sample_lower:
            continue
        if sample_lower in question_lower:
            score += 2.0
        overlap = len(_tokenize(sample_lower) & question_tokens)
        if overlap:
            score += 0.6 * overlap

    if metadata.source_type == "json_row":
        score += 0.25

    return score


async def retrieve_relevant_semantic_context(
    job_id: str,
    table_name: str,
    question: str,
    limit: int = 12,
) -> list[dict[str, Any]]:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(SemanticColumnMetadata).where(
                SemanticColumnMetadata.job_id == job_id,
                SemanticColumnMetadata.table_name == table_name,
            )
        )
        candidates = result.scalars().all()

    if not candidates:
        return []

    question_tokens = _tokenize(question)
    scored = [
        (
            _score_metadata_candidate(question, question_tokens, metadata),
            metadata,
        )
        for metadata in candidates
    ]
    scored.sort(
        key=lambda item: (
            item[0],
            1 if item[1].source_type == "json_row" else 0,
            item[1].column_name.lower(),
        ),
        reverse=True,
    )

    positive_scored = [item for item in scored if item[0] > 0]
    if positive_scored:
        chosen = positive_scored[:limit]
    else:
        chosen = scored[:limit]

    return [
        {
            "column_name": metadata.column_name,
            "data_type": metadata.data_type,
            "source_type": metadata.source_type,
            "description": metadata.description,
            "synonyms": metadata.synonyms or [],
            "sample_values": metadata.sample_values or [],
            "score": round(score, 3),
        }
        for score, metadata in chosen
    ]


def format_semantic_context_for_prompt(context: list[dict[str, Any]]) -> str:
    if not context:
        return ""

    lines = []
    for item in context:
        synonym_preview = ", ".join((item.get("synonyms") or [])[:4]) or "none"
        sample_preview = ", ".join((item.get("sample_values") or [])[:3]) or "none"
        lines.append(
            (
                f"- {item.get('column_name')} [{item.get('source_type')}] "
                f"({item.get('data_type')}), synonyms: {synonym_preview}, "
                f"samples: {sample_preview}, score: {item.get('score')}"
            )
        )
    return "\n".join(lines)
