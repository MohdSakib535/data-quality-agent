"""
SQL Validator using sqlglot AST.

validate_sql() performs 6-step validation:
  A. Parse with sqlglot
  B. AST node type check (reject DML/DDL)
  C. Table scope check (only "dataset" allowed)
  D. LIMIT enforcement (inject/cap)
  E. Subquery check
  F. Reconstruct from AST (never use raw LLM string)
"""
import logging
from dataclasses import dataclass

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from app.core.config import settings

logger = logging.getLogger(__name__)

# Node types that are NEVER allowed in user queries
# sqlglot version differences: some releases expose exp.AlterTable, others only exp.Alter.
# Build the tuple dynamically to avoid AttributeError at import time.
_maybe_alter = getattr(exp, "AlterTable", None) or getattr(exp, "Alter", None)

FORBIDDEN_NODE_TYPES = tuple(
    t
    for t in (
        exp.Insert,
        exp.Update,
        exp.Delete,
        exp.Drop,
        exp.Create,
        _maybe_alter,
        getattr(exp, "Command", None),
    )
    if t is not None
)


@dataclass
class ValidationResult:
    """Result of SQL validation."""
    is_valid: bool
    safe_sql: str | None = None  # AST-reconstructed SQL (safe to execute)
    error: str | None = None
    rejection_reason: str | None = None


def validate_sql(sql: str, dataset_id: str = "") -> ValidationResult:
    """
    Validate LLM-generated SQL using sqlglot AST.
    Returns ValidationResult with safe_sql (reconstructed from AST) if valid.
    NEVER use the original SQL string for execution — only safe_sql.
    """
    sql = sql.strip()
    if not sql:
        return ValidationResult(is_valid=False, error="Empty SQL", rejection_reason="empty_sql")

    # ── Step A: Parse ────────────────────────────────────────
    try:
        parsed = sqlglot.parse_one(sql, dialect="duckdb")
    except ParseError as exc:
        logger.warning(
            "SQL validation: parse failed",
            extra={"dataset_id": dataset_id, "error": str(exc)},
        )
        return ValidationResult(
            is_valid=False,
            error=f"SQL parse error: {str(exc)}",
            rejection_reason="parse_error",
        )

    # ── Step B: AST node type check ──────────────────────────
    for node in parsed.walk():
        if isinstance(node, FORBIDDEN_NODE_TYPES):
            node_type = type(node).__name__
            logger.warning(
                "SQL validation: forbidden node type",
                extra={"dataset_id": dataset_id, "node_type": node_type},
            )
            return ValidationResult(
                is_valid=False,
                error=f"Only SELECT queries are allowed. Found: {node_type}",
                rejection_reason="forbidden_node_type",
            )

    # Ensure the top-level statement is a SELECT
    if not isinstance(parsed, exp.Select):
        return ValidationResult(
            is_valid=False,
            error="Only SELECT queries are allowed",
            rejection_reason="not_select",
        )

    # ── Step C: Table scope check ────────────────────────────
    table_refs = set()
    for table in parsed.find_all(exp.Table):
        table_name = table.name.lower() if table.name else ""
        table_refs.add(table_name)

    invalid_tables = table_refs - {"dataset"}
    if invalid_tables:
        logger.warning(
            "SQL validation: unauthorized table references",
            extra={"dataset_id": dataset_id, "tables": list(invalid_tables)},
        )
        return ValidationResult(
            is_valid=False,
            error=f"Unauthorized table references: {', '.join(invalid_tables)}. Only 'dataset' is allowed.",
            rejection_reason="unauthorized_table",
        )

    if not table_refs:
        return ValidationResult(
            is_valid=False,
            error="No table reference found. Query must reference 'dataset'.",
            rejection_reason="no_table_reference",
        )

    # ── Step D: LIMIT enforcement ────────────────────────────
    limit_node = parsed.find(exp.Limit)
    max_limit = settings.QUERY_HARD_LIMIT

    if limit_node is None:
        # Inject LIMIT
        parsed = parsed.limit(settings.QUERY_MAX_ROWS)
    else:
        # Check if LIMIT is too high
        try:
            limit_val = int(limit_node.expression.this) if hasattr(limit_node.expression, "this") else int(str(limit_node.expression))
            if limit_val > max_limit:
                # Replace with hard limit
                parsed = parsed.limit(max_limit)
        except (ValueError, TypeError, AttributeError):
            # If we can't parse the limit value, cap it
            parsed = parsed.limit(max_limit)

    # ── Step E: Subquery check ───────────────────────────────
    for subquery in parsed.find_all(exp.Subquery):
        # Check table references within subquery
        for table in subquery.find_all(exp.Table):
            sub_table_name = table.name.lower() if table.name else ""
            if sub_table_name and sub_table_name != "dataset":
                logger.warning(
                    "SQL validation: subquery references unauthorized table",
                    extra={"dataset_id": dataset_id, "table": sub_table_name},
                )
                return ValidationResult(
                    is_valid=False,
                    error=f"Subquery references unauthorized table: {sub_table_name}",
                    rejection_reason="subquery_unauthorized_table",
                )

    # ── Step F: Reconstruct from AST ─────────────────────────
    safe_sql = parsed.sql(dialect="duckdb")

    logger.info(
        "SQL validation passed",
        extra={
            "dataset_id": dataset_id,
            "original_sql_hash": hash(sql),
            "safe_sql_length": len(safe_sql),
        },
    )

    return ValidationResult(is_valid=True, safe_sql=safe_sql)
