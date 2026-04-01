"""
SQLAlchemy models: Dataset, Job, DatasetProfile, CleanedDataPreview.
Uses UUID primary keys, proper FK constraints, and JSONB columns.
"""
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    String, BigInteger, Integer, Float, Boolean, Text, DateTime,
    ForeignKey, Index, CheckConstraint,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_uuid() -> uuid.UUID:
    return uuid.uuid4()


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=_new_uuid,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False, index=True)
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    s3_raw_key: Mapped[str] = mapped_column(Text, nullable=False)
    s3_cleaned_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="pending_upload", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)

    # Relationships
    jobs: Mapped[list["Job"]] = relationship(back_populates="dataset", cascade="all, delete-orphan")
    profiles: Mapped[list["DatasetProfile"]] = relationship(back_populates="dataset", cascade="all, delete-orphan")
    previews: Mapped[list["CleanedDataPreview"]] = relationship(back_populates="dataset", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Dataset {self.id} [{self.status}]>"


class Job(Base):
    __tablename__ = "jobs"
    __table_args__ = (
        CheckConstraint(
            "job_type IN ('analyze', 'clean', 'query')",
            name="ck_jobs_job_type",
        ),
        CheckConstraint(
            "status IN ('queued', 'processing', 'completed', 'failed', 'degraded')",
            name="ck_jobs_status",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=_new_uuid,
    )
    dataset_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True,
    )
    job_type: Mapped[str] = mapped_column(String(20), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="queued", index=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # Use a non-reserved attribute name while keeping the DB column name "metadata"
    metadata_json: Mapped[dict] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)

    # Relationships
    dataset: Mapped["Dataset"] = relationship(back_populates="jobs")

    def __repr__(self) -> str:
        return f"<Job {self.id} type={self.job_type} [{self.status}]>"


class DatasetProfile(Base):
    __tablename__ = "dataset_profiles"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=_new_uuid,
    )
    dataset_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True,
    )
    row_count: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    column_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    schema_snapshot: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    null_stats: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    duplicate_count: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    llm_suggestions: Mapped[list | None] = mapped_column(JSONB, default=list)
    profiled_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    # Relationships
    dataset: Mapped["Dataset"] = relationship(back_populates="profiles")

    def __repr__(self) -> str:
        return f"<DatasetProfile {self.id} dataset={self.dataset_id}>"


class CleanedDataPreview(Base):
    __tablename__ = "cleaned_data_preview"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=_new_uuid,
    )
    dataset_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True,
    )
    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True,
    )
    preview_rows: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    cleaned_row_count: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    prompt_used: Mapped[str | None] = mapped_column(Text, nullable=True)
    cleaning_steps_applied: Mapped[list | None] = mapped_column(JSONB, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    # Relationships
    dataset: Mapped["Dataset"] = relationship(back_populates="previews")

    def __repr__(self) -> str:
        return f"<CleanedDataPreview {self.id} job={self.job_id}>"
