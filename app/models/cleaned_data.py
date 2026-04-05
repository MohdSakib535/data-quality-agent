from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, JSON, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class CleanedData(Base):
    __tablename__ = "cleaned_data"

    job_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("jobs.id", ondelete="CASCADE"),
        primary_key=True,
    )
    source_file_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    prompt: Mapped[str] = mapped_column(String, nullable=False)
    cleaned_file_path: Mapped[str] = mapped_column(String, nullable=False)
    cleaned_data: Mapped[list] = mapped_column(JSON, nullable=False)
    cleaned_rows: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    quality_score: Mapped[int] = mapped_column(Integer, nullable=True)
    analysis: Mapped[dict] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )
