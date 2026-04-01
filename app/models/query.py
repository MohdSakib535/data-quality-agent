"""
SQLAlchemy models: QueryLog, ParquetSchemaVersion.
"""
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    String, Integer, Float, Boolean, Text, DateTime,
    ForeignKey,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_uuid() -> uuid.UUID:
    return uuid.uuid4()


class QueryLog(Base):
    __tablename__ = "query_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=_new_uuid,
    )
    dataset_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True,
    )
    natural_language_query: Mapped[str] = mapped_column(Text, nullable=False)
    generated_sql: Mapped[str | None] = mapped_column(Text, nullable=True)
    validated: Mapped[bool] = mapped_column(Boolean, default=False)
    confidence_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    execution_time_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    result_row_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)

    def __repr__(self) -> str:
        return f"<QueryLog {self.id} dataset={self.dataset_id}>"


class ParquetSchemaVersion(Base):
    __tablename__ = "parquet_schema_versions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=_new_uuid,
    )
    dataset_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True,
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    schema_fingerprint: Mapped[str] = mapped_column(Text, nullable=False)
    column_map: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
    parquet_metadata: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    def __repr__(self) -> str:
        return f"<ParquetSchemaVersion {self.id} v{self.version} dataset={self.dataset_id}>"
