"""Baseline schema migration.

Revision ID: 20260411_0001
Revises:
Create Date: 2026-04-11 00:00:00.000000

"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "20260411_0001"
down_revision = None
branch_labels = None
depends_on = None


def _inspector():
    return sa.inspect(op.get_bind())


def _has_table(table_name: str) -> bool:
    return table_name in _inspector().get_table_names()


def _has_column(table_name: str, column_name: str) -> bool:
    return any(column["name"] == column_name for column in _inspector().get_columns(table_name))


def _has_index(table_name: str, index_name: str) -> bool:
    return any(index["name"] == index_name for index in _inspector().get_indexes(table_name))


def _has_unique_constraint(table_name: str, constraint_name: str) -> bool:
    return any(
        constraint["name"] == constraint_name
        for constraint in _inspector().get_unique_constraints(table_name)
    )


def _ensure_jobs_table() -> None:
    if not _has_table("jobs"):
        op.create_table(
            "jobs",
            sa.Column("id", sa.String(), nullable=False),
            sa.Column("status", sa.String(), nullable=False),
            sa.Column("filename", sa.String(), nullable=True),
            sa.Column("file_url", sa.String(), nullable=True),
            sa.Column("message", sa.String(), nullable=True),
            sa.Column("quality_score", sa.Integer(), nullable=True),
            sa.Column("analysis", sa.JSON(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=True),
            sa.Column("updated_at", sa.DateTime(), nullable=True),
            sa.PrimaryKeyConstraint("id"),
        )

    if not _has_column("jobs", "status"):
        op.add_column("jobs", sa.Column("status", sa.String(), nullable=False, server_default="pending"))
        op.alter_column("jobs", "status", existing_type=sa.String(), server_default=None)
    if not _has_column("jobs", "filename"):
        op.add_column("jobs", sa.Column("filename", sa.String(), nullable=True))
    if not _has_column("jobs", "file_url"):
        op.add_column("jobs", sa.Column("file_url", sa.String(), nullable=True))
    if not _has_column("jobs", "message"):
        op.add_column("jobs", sa.Column("message", sa.String(), nullable=True))
    if not _has_column("jobs", "quality_score"):
        op.add_column("jobs", sa.Column("quality_score", sa.Integer(), nullable=True))
    if not _has_column("jobs", "analysis"):
        op.add_column("jobs", sa.Column("analysis", sa.JSON(), nullable=True))
    if not _has_column("jobs", "created_at"):
        op.add_column("jobs", sa.Column("created_at", sa.DateTime(), nullable=True))
    if not _has_column("jobs", "updated_at"):
        op.add_column("jobs", sa.Column("updated_at", sa.DateTime(), nullable=True))

    if not _has_index("jobs", "ix_jobs_status"):
        op.create_index("ix_jobs_status", "jobs", ["status"], unique=False)


def _ensure_cleaned_data_table() -> None:
    if not _has_table("cleaned_data"):
        op.create_table(
            "cleaned_data",
            sa.Column("job_id", sa.String(), nullable=False),
            sa.Column("source_file_id", sa.String(), nullable=False),
            sa.Column("prompt", sa.String(), nullable=False),
            sa.Column("source_type", sa.String(), nullable=False),
            sa.Column("cleaning_strategy", sa.String(), nullable=True),
            sa.Column("target_columns", sa.JSON(), nullable=False),
            sa.Column("cleaned_file_path", sa.String(), nullable=False),
            sa.Column("cleaned_data", sa.JSON(), nullable=False),
            sa.Column("cleaned_rows", sa.Integer(), nullable=False),
            sa.Column("preview_rows_returned", sa.Integer(), nullable=False),
            sa.Column("preview_limited", sa.Boolean(), nullable=False),
            sa.Column("changes_detected", sa.Boolean(), nullable=False),
            sa.Column("quality_score", sa.Integer(), nullable=True),
            sa.Column("analysis", sa.JSON(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=True),
            sa.Column("updated_at", sa.DateTime(), nullable=True),
            sa.ForeignKeyConstraint(["job_id"], ["jobs.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("job_id"),
        )

    if not _has_column("cleaned_data", "source_file_id"):
        op.add_column("cleaned_data", sa.Column("source_file_id", sa.String(), nullable=True))
    op.execute("UPDATE cleaned_data SET source_file_id = job_id WHERE source_file_id IS NULL")
    op.alter_column("cleaned_data", "source_file_id", existing_type=sa.String(), nullable=False)

    if not _has_column("cleaned_data", "prompt"):
        op.add_column("cleaned_data", sa.Column("prompt", sa.String(), nullable=True))

    if not _has_column("cleaned_data", "source_type"):
        op.add_column("cleaned_data", sa.Column("source_type", sa.String(), nullable=True))
    op.execute("UPDATE cleaned_data SET source_type = 'raw' WHERE source_type IS NULL")
    op.alter_column("cleaned_data", "source_type", existing_type=sa.String(), nullable=False)

    if not _has_column("cleaned_data", "cleaning_strategy"):
        op.add_column("cleaned_data", sa.Column("cleaning_strategy", sa.String(), nullable=True))

    if not _has_column("cleaned_data", "target_columns"):
        op.add_column("cleaned_data", sa.Column("target_columns", sa.JSON(), nullable=True))
    op.execute("UPDATE cleaned_data SET target_columns = '[]'::json WHERE target_columns IS NULL")
    op.alter_column("cleaned_data", "target_columns", existing_type=sa.JSON(), nullable=False)

    if not _has_column("cleaned_data", "cleaned_file_path"):
        op.add_column("cleaned_data", sa.Column("cleaned_file_path", sa.String(), nullable=True))

    if not _has_column("cleaned_data", "cleaned_data"):
        op.add_column("cleaned_data", sa.Column("cleaned_data", sa.JSON(), nullable=True))

    if not _has_column("cleaned_data", "cleaned_rows"):
        op.add_column("cleaned_data", sa.Column("cleaned_rows", sa.Integer(), nullable=True))
    op.execute("UPDATE cleaned_data SET cleaned_rows = 0 WHERE cleaned_rows IS NULL")
    op.alter_column("cleaned_data", "cleaned_rows", existing_type=sa.Integer(), nullable=False)

    if not _has_column("cleaned_data", "preview_rows_returned"):
        op.add_column("cleaned_data", sa.Column("preview_rows_returned", sa.Integer(), nullable=True))
    op.execute("UPDATE cleaned_data SET preview_rows_returned = 0 WHERE preview_rows_returned IS NULL")
    op.alter_column("cleaned_data", "preview_rows_returned", existing_type=sa.Integer(), nullable=False)

    if not _has_column("cleaned_data", "preview_limited"):
        op.add_column("cleaned_data", sa.Column("preview_limited", sa.Boolean(), nullable=True))
    op.execute("UPDATE cleaned_data SET preview_limited = FALSE WHERE preview_limited IS NULL")
    op.alter_column("cleaned_data", "preview_limited", existing_type=sa.Boolean(), nullable=False)

    if not _has_column("cleaned_data", "changes_detected"):
        op.add_column("cleaned_data", sa.Column("changes_detected", sa.Boolean(), nullable=True))
    op.execute("UPDATE cleaned_data SET changes_detected = FALSE WHERE changes_detected IS NULL")
    op.alter_column("cleaned_data", "changes_detected", existing_type=sa.Boolean(), nullable=False)

    if not _has_column("cleaned_data", "quality_score"):
        op.add_column("cleaned_data", sa.Column("quality_score", sa.Integer(), nullable=True))

    if not _has_column("cleaned_data", "analysis"):
        op.add_column("cleaned_data", sa.Column("analysis", sa.JSON(), nullable=True))

    if not _has_column("cleaned_data", "created_at"):
        op.add_column("cleaned_data", sa.Column("created_at", sa.DateTime(), nullable=True))
    if not _has_column("cleaned_data", "updated_at"):
        op.add_column("cleaned_data", sa.Column("updated_at", sa.DateTime(), nullable=True))

    if not _has_index("cleaned_data", "ix_cleaned_data_source_file_id"):
        op.create_index("ix_cleaned_data_source_file_id", "cleaned_data", ["source_file_id"], unique=False)


def _ensure_analysis_suggestions_table() -> None:
    if not _has_table("analysis_suggestions"):
        op.create_table(
            "analysis_suggestions",
            sa.Column("id", sa.String(), nullable=False),
            sa.Column("job_id", sa.String(), nullable=False),
            sa.Column("source_type", sa.String(), nullable=False),
            sa.Column("issue_description", sa.Text(), nullable=False),
            sa.Column("priority", sa.String(), nullable=False),
            sa.Column("resolution_prompt", sa.Text(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=True),
            sa.Column("updated_at", sa.DateTime(), nullable=True),
            sa.ForeignKeyConstraint(["job_id"], ["jobs.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
        )

    if not _has_column("analysis_suggestions", "job_id"):
        op.add_column("analysis_suggestions", sa.Column("job_id", sa.String(), nullable=True))
    if not _has_column("analysis_suggestions", "source_type"):
        op.add_column("analysis_suggestions", sa.Column("source_type", sa.String(), nullable=True))
    op.execute("UPDATE analysis_suggestions SET source_type = 'raw' WHERE source_type IS NULL")
    op.alter_column("analysis_suggestions", "source_type", existing_type=sa.String(), nullable=False)

    if not _has_column("analysis_suggestions", "issue_description"):
        op.add_column("analysis_suggestions", sa.Column("issue_description", sa.Text(), nullable=True))
    if not _has_column("analysis_suggestions", "priority"):
        op.add_column("analysis_suggestions", sa.Column("priority", sa.String(), nullable=True))
    if not _has_column("analysis_suggestions", "resolution_prompt"):
        op.add_column("analysis_suggestions", sa.Column("resolution_prompt", sa.Text(), nullable=True))
    if not _has_column("analysis_suggestions", "created_at"):
        op.add_column("analysis_suggestions", sa.Column("created_at", sa.DateTime(), nullable=True))
    if not _has_column("analysis_suggestions", "updated_at"):
        op.add_column("analysis_suggestions", sa.Column("updated_at", sa.DateTime(), nullable=True))

    if not _has_index("analysis_suggestions", "ix_analysis_suggestions_job_id"):
        op.create_index("ix_analysis_suggestions_job_id", "analysis_suggestions", ["job_id"], unique=False)
    if not _has_index("analysis_suggestions", "ix_analysis_suggestions_source_type"):
        op.create_index("ix_analysis_suggestions_source_type", "analysis_suggestions", ["source_type"], unique=False)


def _ensure_chat_history_table() -> None:
    if not _has_table("chat_history"):
        op.create_table(
            "chat_history",
            sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
            sa.Column("job_id", sa.String(length=255), nullable=False),
            sa.Column("question", sa.Text(), nullable=False),
            sa.Column("sql_generated", sa.Text(), nullable=True),
            sa.Column("row_count", sa.Integer(), nullable=True),
            sa.Column("success", sa.Boolean(), nullable=False),
            sa.Column("error_message", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=True),
            sa.PrimaryKeyConstraint("id"),
        )

    if not _has_column("chat_history", "job_id"):
        op.add_column("chat_history", sa.Column("job_id", sa.String(length=255), nullable=True))
    if not _has_column("chat_history", "question"):
        op.add_column("chat_history", sa.Column("question", sa.Text(), nullable=True))
    if not _has_column("chat_history", "sql_generated"):
        op.add_column("chat_history", sa.Column("sql_generated", sa.Text(), nullable=True))
    if not _has_column("chat_history", "row_count"):
        op.add_column("chat_history", sa.Column("row_count", sa.Integer(), nullable=True))
    if not _has_column("chat_history", "success"):
        op.add_column("chat_history", sa.Column("success", sa.Boolean(), nullable=True))
    op.execute("UPDATE chat_history SET success = TRUE WHERE success IS NULL")
    op.alter_column("chat_history", "success", existing_type=sa.Boolean(), nullable=False)
    if not _has_column("chat_history", "error_message"):
        op.add_column("chat_history", sa.Column("error_message", sa.Text(), nullable=True))
    if not _has_column("chat_history", "created_at"):
        op.add_column("chat_history", sa.Column("created_at", sa.DateTime(), nullable=True))

    if not _has_index("chat_history", "ix_chat_history_job_id"):
        op.create_index("ix_chat_history_job_id", "chat_history", ["job_id"], unique=False)
    if not _has_index("chat_history", "ix_chat_history_created_at"):
        op.create_index("ix_chat_history_created_at", "chat_history", ["created_at"], unique=False)


def _ensure_semantic_column_metadata_table() -> None:
    if not _has_table("semantic_column_metadata"):
        op.create_table(
            "semantic_column_metadata",
            sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
            sa.Column("job_id", sa.String(length=255), nullable=False),
            sa.Column("table_name", sa.String(length=255), nullable=False),
            sa.Column("source_type", sa.String(length=50), nullable=False),
            sa.Column("column_name", sa.String(length=255), nullable=False),
            sa.Column("data_type", sa.String(length=255), nullable=False),
            sa.Column("description", sa.Text(), nullable=True),
            sa.Column("synonyms", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
            sa.Column("sample_values", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=True),
            sa.Column("updated_at", sa.DateTime(), nullable=True),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint(
                "job_id",
                "table_name",
                "source_type",
                "column_name",
                name="uq_semantic_column_metadata_job_table_source_col",
            ),
        )

    if not _has_column("semantic_column_metadata", "job_id"):
        op.add_column("semantic_column_metadata", sa.Column("job_id", sa.String(length=255), nullable=True))
    if not _has_column("semantic_column_metadata", "table_name"):
        op.add_column("semantic_column_metadata", sa.Column("table_name", sa.String(length=255), nullable=True))
    if not _has_column("semantic_column_metadata", "source_type"):
        op.add_column("semantic_column_metadata", sa.Column("source_type", sa.String(length=50), nullable=True))
    if not _has_column("semantic_column_metadata", "column_name"):
        op.add_column("semantic_column_metadata", sa.Column("column_name", sa.String(length=255), nullable=True))
    if not _has_column("semantic_column_metadata", "data_type"):
        op.add_column("semantic_column_metadata", sa.Column("data_type", sa.String(length=255), nullable=True))
    if not _has_column("semantic_column_metadata", "description"):
        op.add_column("semantic_column_metadata", sa.Column("description", sa.Text(), nullable=True))
    if not _has_column("semantic_column_metadata", "synonyms"):
        op.add_column(
            "semantic_column_metadata",
            sa.Column("synonyms", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        )
    op.execute("UPDATE semantic_column_metadata SET synonyms = '[]'::jsonb WHERE synonyms IS NULL")
    op.alter_column(
        "semantic_column_metadata",
        "synonyms",
        existing_type=postgresql.JSONB(astext_type=sa.Text()),
        nullable=False,
    )
    if not _has_column("semantic_column_metadata", "sample_values"):
        op.add_column(
            "semantic_column_metadata",
            sa.Column("sample_values", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        )
    op.execute("UPDATE semantic_column_metadata SET sample_values = '[]'::jsonb WHERE sample_values IS NULL")
    op.alter_column(
        "semantic_column_metadata",
        "sample_values",
        existing_type=postgresql.JSONB(astext_type=sa.Text()),
        nullable=False,
    )
    if not _has_column("semantic_column_metadata", "created_at"):
        op.add_column("semantic_column_metadata", sa.Column("created_at", sa.DateTime(), nullable=True))
    if not _has_column("semantic_column_metadata", "updated_at"):
        op.add_column("semantic_column_metadata", sa.Column("updated_at", sa.DateTime(), nullable=True))

    if not _has_index("semantic_column_metadata", "ix_semantic_column_metadata_job_id"):
        op.create_index("ix_semantic_column_metadata_job_id", "semantic_column_metadata", ["job_id"], unique=False)
    if not _has_index("semantic_column_metadata", "ix_semantic_column_metadata_table_name"):
        op.create_index("ix_semantic_column_metadata_table_name", "semantic_column_metadata", ["table_name"], unique=False)
    if not _has_index("semantic_column_metadata", "ix_semantic_column_metadata_source_type"):
        op.create_index("ix_semantic_column_metadata_source_type", "semantic_column_metadata", ["source_type"], unique=False)
    if not _has_index("semantic_column_metadata", "ix_semantic_column_metadata_column_name"):
        op.create_index("ix_semantic_column_metadata_column_name", "semantic_column_metadata", ["column_name"], unique=False)
    if not _has_unique_constraint(
        "semantic_column_metadata",
        "uq_semantic_column_metadata_job_table_source_col",
    ):
        op.create_unique_constraint(
            "uq_semantic_column_metadata_job_table_source_col",
            "semantic_column_metadata",
            ["job_id", "table_name", "source_type", "column_name"],
        )


def upgrade() -> None:
    _ensure_jobs_table()
    _ensure_cleaned_data_table()
    _ensure_analysis_suggestions_table()
    _ensure_chat_history_table()
    _ensure_semantic_column_metadata_table()


def downgrade() -> None:
    if _has_table("semantic_column_metadata"):
        op.drop_table("semantic_column_metadata")
    if _has_table("chat_history"):
        op.drop_table("chat_history")
    if _has_table("analysis_suggestions"):
        op.drop_table("analysis_suggestions")
    if _has_table("cleaned_data"):
        op.drop_table("cleaned_data")
    if _has_table("jobs"):
        op.drop_table("jobs")
