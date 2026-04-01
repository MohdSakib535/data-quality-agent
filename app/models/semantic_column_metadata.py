from datetime import datetime

from sqlalchemy import DateTime, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class SemanticColumnMetadata(Base):
    __tablename__ = "semantic_column_metadata"
    __table_args__ = (
        UniqueConstraint(
            "job_id",
            "table_name",
            "source_type",
            "column_name",
            name="uq_semantic_column_metadata_job_table_source_col",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    table_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    source_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    column_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    data_type: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=True)
    synonyms: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    sample_values: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )
