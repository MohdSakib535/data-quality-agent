"""
Application configuration via pydantic-settings.
All values from environment variables — never hardcoded secrets.
"""
from pydantic_settings import BaseSettings
from pydantic import ConfigDict


class Settings(BaseSettings):
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ──────────────────────────────────────────────
    PROJECT_NAME: str = "Dataset Intelligence Platform"
    VERSION: str = "2.0.0"
    DEBUG: bool = False

    # ── Database (async for API) ─────────────────────────
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/ai_cleaning_db1"
    SYNC_DATABASE_URL: str = "postgresql+psycopg2://postgres:postgres@localhost:5432/ai_cleaning_db1"

    # ── Redis ────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"

    # ── Celery ───────────────────────────────────────────
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"

    # ── S3 / MinIO ───────────────────────────────────────
    S3_ENDPOINT_URL: str = "http://localhost:9000"
    S3_ACCESS_KEY_ID: str = "minioadmin"
    S3_SECRET_ACCESS_KEY: str = "minioadmin"
    S3_BUCKET_NAME: str = "datasets"
    S3_REGION: str = "us-east-1"
    S3_PRESIGNED_URL_TTL_UPLOAD: int = 900      # 15 minutes
    S3_PRESIGNED_URL_TTL_DOWNLOAD: int = 3600   # 1 hour

    # ── LLM (Ollama) ─────────────────────────────────────
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2.5-coder"
    LLM_PROVIDER: str = "ollama"  # "ollama" | "openai" | "anthropic"
    LLM_TIMEOUT: int = 60
    LLM_CACHE_TTL: int = 86400  # 24 hours
    # Optional cloud LLM fallback keys (leave empty if not used)
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o"
    ANTHROPIC_API_KEY: str = ""
    ANTHROPIC_MODEL: str = "claude-sonnet-4-20250514"

    # ── Processing ───────────────────────────────────────
    CHUNK_SIZE: int = 1000
    AI_BATCH_SIZE: int = 50
    ANALYSIS_SAMPLE_ROWS: int = 5000
    MAX_FILE_SIZE_BYTES: int = 524_288_000  # 500 MB
    CLEAN_PREVIEW_ROWS: int = 25
    QUERY_TIMEOUT: int = 30
    QUERY_MAX_ROWS: int = 1000
    QUERY_HARD_LIMIT: int = 10000
    DUCKDB_MAX_CONNECTIONS: int = 10

    # ── Allowed extensions ───────────────────────────────
    ALLOWED_EXTENSIONS: list[str] = [".csv", ".xlsx", ".xls"]
    FILE_SIZE_TOLERANCE: float = 0.05  # ±5%


settings = Settings()
