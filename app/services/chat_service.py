import json
import logging
import re
from typing import Any

import httpx
import redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.db.session import AsyncSessionLocal, engine
from app.models.chat_history import ChatHistory
from app.utils.chat_utils import (
    OllamaResponseError,
    OllamaUnavailableError,
    clean_sql_response,
)

logger = logging.getLogger(__name__)

SCHEMA_CACHE_TTL_SECONDS = 300


def resolve_ollama_host() -> str:
    configured_host = (settings.OLLAMA_HOST or "").strip().rstrip("/")
    if configured_host and "your-ollama-server" not in configured_host:
        return configured_host
    return settings.OLLAMA_BASE_URL.rstrip("/")


def _redis_client() -> redis.Redis | None:
    try:
        client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            decode_responses=True,
            socket_connect_timeout=0.5,
            socket_timeout=0.5,
        )
        client.ping()
        return client
    except Exception:
        logger.warning(
            "Redis unavailable at %s:%s; schema cache disabled.",
            settings.REDIS_HOST,
            settings.REDIS_PORT,
        )
        return None


async def get_table_schema(table_name: str) -> list[dict]:
    cache_key = f"schema:{table_name}"
    redis_client = _redis_client()

    if redis_client is not None:
        try:
            cached_schema = redis_client.get(cache_key)
            if cached_schema:
                parsed = json.loads(cached_schema)
                if isinstance(parsed, list):
                    return parsed
        except Exception:
            logger.warning("Redis cache read failed for key=%s", cache_key)

    query = text(
        """
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = :t
          AND table_schema = 'public'
        ORDER BY ordinal_position
        """
    )

    async with engine.connect() as conn:
        result = await conn.execute(query, {"t": table_name})
        rows = result.fetchmany(1000)

    schema = [
        {"column_name": row.column_name, "data_type": row.data_type}
        for row in rows
    ]

    if redis_client is not None:
        try:
            redis_client.setex(cache_key, SCHEMA_CACHE_TTL_SECONDS, json.dumps(schema))
        except Exception:
            logger.warning("Redis cache write failed for key=%s", cache_key)

    return schema


def _infer_value_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "numeric"
    if value is None:
        return "null"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "text"


async def get_cleaned_data_row_schema(job_id: str, sample_items: int = 200) -> list[dict]:
    query = text(
        """
        SELECT cleaned_data
        FROM cleaned_data
        WHERE job_id = :job_id
        LIMIT 1
        """
    )

    async with engine.connect() as conn:
        result = await conn.execute(query, {"job_id": job_id})
        row = result.mappings().fetchone()

    if not row:
        return []

    payload = row.get("cleaned_data")
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except ValueError:
            return []

    records: list[dict[str, Any]] = []
    if isinstance(payload, list):
        for item in payload[:sample_items]:
            if isinstance(item, dict):
                records.append(item)
    elif isinstance(payload, dict):
        records = [payload]
    else:
        return []

    field_types: dict[str, set[str]] = {}
    for record in records:
        for key, value in record.items():
            inferred = _infer_value_type(value)
            if key not in field_types:
                field_types[key] = set()
            field_types[key].add(inferred)

    return [
        {"column_name": key, "data_type": "/".join(sorted(types))}
        for key, types in sorted(field_types.items(), key=lambda item: item[0])
    ]


def _extract_select_expressions(sql: str) -> list[str]:
    select_match = re.search(r"\bSELECT\b", sql, flags=re.IGNORECASE)
    if not select_match:
        return []

    start = select_match.end()
    depth = 0
    in_single = False
    in_double = False
    from_pos = -1
    i = start

    while i < len(sql):
        ch = sql[i]

        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif not in_single and not in_double:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth = max(0, depth - 1)
            elif (
                depth == 0
                and sql[i:i + 4].upper() == "FROM"
                and (i == 0 or not (sql[i - 1].isalnum() or sql[i - 1] == "_"))
                and (i + 4 >= len(sql) or not (sql[i + 4].isalnum() or sql[i + 4] == "_"))
            ):
                from_pos = i
                break
        i += 1

    if from_pos == -1:
        return []

    select_clause = sql[start:from_pos].strip()
    if not select_clause:
        return []

    expressions: list[str] = []
    token = []
    depth = 0
    in_single = False
    in_double = False

    for ch in select_clause:
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif not in_single and not in_double:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth = max(0, depth - 1)
            elif ch == "," and depth == 0:
                expr = "".join(token).strip()
                if expr:
                    expressions.append(expr)
                token = []
                continue
        token.append(ch)

    tail = "".join(token).strip()
    if tail:
        expressions.append(tail)

    return expressions


def _infer_expression_alias(expression: str, index: int) -> str:
    as_match = re.search(r"\bAS\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*$", expression, flags=re.IGNORECASE)
    if as_match:
        return as_match.group(1)

    json_key_match = re.search(r"->>\s*'([a-zA-Z_][a-zA-Z0-9_]*)'", expression)
    if json_key_match:
        return json_key_match.group(1)

    dotted_col_match = re.search(r"(?:^|\.)([a-zA-Z_][a-zA-Z0-9_]*)\s*$", expression.strip())
    if dotted_col_match and "(" not in expression:
        return dotted_col_match.group(1)

    func_match = re.match(r"\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", expression)
    if func_match:
        return func_match.group(1).lower()

    return f"column_{index + 1}"


def _normalized_result_keys(result_keys: list[str], sql: str) -> list[str]:
    expressions = _extract_select_expressions(sql)
    normalized: list[str] = []
    used: dict[str, int] = {}

    for idx, key in enumerate(result_keys):
        candidate = key
        if key.startswith("?column?"):
            if idx < len(expressions):
                candidate = _infer_expression_alias(expressions[idx], idx)
            else:
                candidate = f"column_{idx + 1}"

        base = candidate
        count = used.get(base, 0) + 1
        used[base] = count
        if count > 1:
            candidate = f"{base}_{count}"
        normalized.append(candidate)

    return normalized


def build_prompt(
    question: str,
    schema: list[dict],
    table_name: str,
    row_schema: list[dict] | None = None,
) -> str:
    columns = "\n".join(
        f"- {column['column_name']} ({column['data_type']})"
        for column in schema
    )

    extra_rules = ""
    if table_name == "cleaned_data":
        json_columns = "\n".join(
            f"- {column['column_name']} ({column['data_type']})"
            for column in (row_schema or [])
        )
        extra_rules = """
Special rules for cleaned_data:
- cleaned_data is a JSON array of row objects (not a flat relational table).
- Use CROSS JOIN LATERAL jsonb_array_elements(cleaned_data::jsonb) AS row(item) to expand rows.
- Read JSON fields with item->>'field_name' (text extraction).
- Cast for numeric ops with NULLIF(item->>'field_name','')::numeric.
- Do not use json_array_elements_text(...)->... because -> cannot be used on text.
- Do not return prompt or cleaned_file_path unless explicitly requested.
- For analytical questions, always query expanded JSON rows, not the top-level cleaned_data table row.
- Preferred query shape:
  SELECT ... FROM cleaned_data cd
  CROSS JOIN LATERAL jsonb_array_elements(cd.cleaned_data::jsonb) AS row(item)
  WHERE cd.job_id = '<job_id>'
"""
        if json_columns:
            extra_rules += f"""
Detected JSON row fields:
{json_columns}
"""

    return f"""You are a PostgreSQL expert. Write a SELECT query only.

Table: {table_name}
Columns:
{columns}
{extra_rules}

Rules:
- Return ONLY raw SQL, no explanation, no markdown, no backticks
- Only SELECT statements
- Use exact column names listed above
- Add explicit aliases for every selected expression using AS
- Add LIMIT 100 unless question asks for specific count
- If columns are insufficient write:
  SELECT 'Cannot answer: insufficient columns' as message

Question: {question}

SQL:
"""


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=4),
    reraise=True,
)
async def _call_ollama(prompt: str) -> str:
    ollama_host = resolve_ollama_host()
    ollama_model = settings.OLLAMA_MODEL
    ollama_timeout = float(settings.OLLAMA_TIMEOUT)
    endpoint = f"{ollama_host}/api/generate"

    payload = {
        "model": ollama_model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0, "num_predict": 300},
    }

    try:
        async with httpx.AsyncClient(timeout=ollama_timeout) as client:
            response = await client.post(endpoint, json=payload)
    except (httpx.ConnectError, httpx.TimeoutException) as exc:
        raise OllamaUnavailableError("Ollama is unavailable.") from exc
    except httpx.HTTPError as exc:
        raise OllamaResponseError("Failed to call Ollama.") from exc

    if response.status_code >= 500:
        raise OllamaUnavailableError("Ollama server returned an error.")
    if response.status_code >= 400:
        raise OllamaResponseError("Ollama returned an invalid response.")

    try:
        raw_sql = response.json()["response"]
    except (ValueError, KeyError, TypeError) as exc:
        raise OllamaResponseError("Invalid Ollama response format.") from exc

    sql = clean_sql_response(raw_sql)
    if not sql:
        raise OllamaResponseError("Ollama did not return SQL.")

    return sql


async def generate_sql_via_ollama(
    question: str,
    schema: list[dict],
    table_name: str,
    row_schema: list[dict] | None = None,
) -> str:
    prompt = build_prompt(question, schema, table_name, row_schema=row_schema)
    return await _call_ollama(prompt)


def build_repair_prompt(
    question: str,
    schema: list[dict],
    table_name: str,
    failed_sql: str,
    db_error: str,
    row_schema: list[dict] | None = None,
) -> str:
    columns = "\n".join(
        f"- {column['column_name']} ({column['data_type']})"
        for column in schema
    )
    json_rules = ""
    if table_name == "cleaned_data":
        json_rules = """
For cleaned_data table:
- cleaned_data is a JSON array of row objects.
- Use jsonb_array_elements(cleaned_data::jsonb) AS row(item) to expand rows.
- Extract values with item->>'field_name'.
- Cast numeric fields explicitly before aggregation.
"""
        if row_schema:
            detected = "\n".join(
                f"- {column['column_name']} ({column['data_type']})"
                for column in row_schema
            )
            json_rules += f"""
Detected JSON row fields:
{detected}
"""

    return f"""You are a PostgreSQL expert. Fix this SQL query.

Table: {table_name}
Columns:
{columns}
{json_rules}

Original question:
{question}

Failed SQL:
{failed_sql}

Database error:
{db_error}

Rules:
- Return ONLY corrected raw SQL, no explanation, no markdown, no backticks
- Only SELECT statements
- Keep the same intent as the question
- Use valid PostgreSQL syntax
- Add explicit aliases for every selected expression using AS
- Add LIMIT 100 unless the query already has LIMIT or asks for count only

SQL:
"""


async def repair_sql_via_ollama(
    question: str,
    schema: list[dict],
    table_name: str,
    failed_sql: str,
    db_error: str,
    row_schema: list[dict] | None = None,
) -> str:
    prompt = build_repair_prompt(
        question,
        schema,
        table_name,
        failed_sql,
        db_error,
        row_schema=row_schema,
    )
    return await _call_ollama(prompt)


async def safe_execute(conn: AsyncConnection, sql: str, max_rows: int = 500) -> list[dict]:
    normalized_sql = sql.strip().rstrip(";")
    if "LIMIT" not in normalized_sql.upper():
        normalized_sql = f"{normalized_sql} LIMIT {max_rows}"

    result = await conn.execute(text(normalized_sql))
    rows = result.fetchmany(max_rows)
    result_keys = list(result.keys())
    normalized_keys = _normalized_result_keys(result_keys, normalized_sql)

    payload: list[dict] = []
    for row in rows:
        row_values = tuple(row)
        payload.append(
            {
                normalized_keys[i]: row_values[i]
                for i in range(min(len(normalized_keys), len(row_values)))
            }
        )
    return payload


async def save_chat_history(
    job_id: str,
    question: str,
    sql_generated: str,
    row_count: int,
    success: bool,
    error_message: str | None = None,
) -> None:
    try:
        async with AsyncSessionLocal() as session:
            history = ChatHistory(
                job_id=job_id,
                question=question,
                sql_generated=sql_generated,
                row_count=row_count,
                success=success,
                error_message=error_message,
            )
            session.add(history)
            await session.commit()
    except Exception:
        logger.exception("Failed to save chat history for job_id=%s", job_id)
