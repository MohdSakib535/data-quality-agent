import re
import time


class OllamaUnavailableError(Exception):
    pass


class OllamaResponseError(Exception):
    pass


class JobNotFoundError(Exception):
    pass


class InvalidSQLError(Exception):
    pass


def validate_sql(sql: str) -> tuple[bool, str]:
    if not sql or not isinstance(sql, str):
        return False, "SQL is empty."

    normalized = sql.strip()
    upper_sql = normalized.upper()

    if not upper_sql.startswith("SELECT"):
        return False, "Only SELECT statements are allowed."

    blocked_keywords = [
        "DROP",
        "DELETE",
        "UPDATE",
        "INSERT",
        "ALTER",
        "TRUNCATE",
        "CREATE",
        "EXEC",
        "GRANT",
        "REVOKE",
        "COPY",
        "VACUUM",
    ]

    for keyword in blocked_keywords:
        if re.search(rf"\b{keyword}\b", upper_sql):
            return False, f"Blocked SQL keyword detected: {keyword}."

    blocked_patterns = [r"--", r"/\*", r"\*/", r";"]
    for pattern in blocked_patterns:
        if re.search(pattern, normalized):
            return False, f"Blocked SQL pattern detected: {pattern}."

    return True, "valid"


def sanitize_question(question: str) -> str:
    cleaned = (question or "").replace("\x00", "")
    cleaned = cleaned.replace("```", "").strip()
    return cleaned[:500]


def clean_sql_response(raw: str) -> str:
    if not raw or not isinstance(raw, str):
        return ""

    cleaned = re.sub(r"```sql", "", raw, flags=re.IGNORECASE)
    cleaned = cleaned.replace("```", "")

    select_match = re.search(r"\bSELECT\b", cleaned, flags=re.IGNORECASE)
    if not select_match:
        return ""

    sql = cleaned[select_match.start():].strip()
    return sql.rstrip(";").strip()


def format_execution_time(start: float) -> float:
    return round((time.time() - start) * 1000, 2)
