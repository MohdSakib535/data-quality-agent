from typing import Optional
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    PROJECT_NAME: str = "AI CSV Cleaning Agent"
    VERSION: str = "1.0.0"
    
    # Storage settings
    S3_ENDPOINT_URL: Optional[str] = None
    S3_ACCESS_KEY_ID: Optional[str] = None
    S3_SECRET_ACCESS_KEY: Optional[str] = None
    S3_BUCKET_NAME: Optional[str] = None
    S3_REGION: str = "us-east-1"
    
    # LLM Settings
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2.5:7b-instruct"
    OLLAMA_HOST: Optional[str] = None
    OLLAMA_TIMEOUT: int = 30
    
    # Database settings
    DATABASE_URL: str
    
    # Processing settings
    CHUNK_SIZE: int = 1000 # For processing large CSVs chunk by chunk
    AI_BATCH_SIZE: int = 50
    AI_VALUE_BATCH_SIZE: int = 200
    AI_VALUE_CLEAN_MAX_UNIQUE_VALUES: int = 5000
    AI_VALUE_CLEAN_MAX_COLUMNS: int = 5
    AI_ROW_LEVEL_LARGE_DATASET_THRESHOLD: int = 50000
    ANALYSIS_SAMPLE_ROWS: int = 500
    CLEAN_PREVIEW_ROWS: int = 25
    CONFIDENCE_AUTO_ACCEPT: float = 0.95
    CONFIDENCE_REVIEW: float = 0.75
    DATE_OUTPUT_FORMAT: str = "%Y-%m-%d"
    DATETIME_OUTPUT_FORMAT: str = "%Y-%m-%dT%H:%M:%S"
    PHONE_DEFAULT_COUNTRY_CODE: Optional[str] = None
    NORMALIZE_PERCENTAGE_TO_FRACTION: bool = False
    NULL_TOKENS: str = "n/a,na,null,none,-,--,,n.a.,nil,missing,not available,not provided,blank,#n/a,nan,tbd"
    NULL_OUTPUT_TOKEN: str = "N/A"
    CHAT_MAX_ROWS: int = 500
    SEMANTIC_TOP_COLUMNS: int = 12
    SEMANTIC_ROW_SAMPLE_LIMIT: int = 200
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0

    @field_validator(
        "S3_ENDPOINT_URL",
        "S3_ACCESS_KEY_ID",
        "S3_SECRET_ACCESS_KEY",
        "S3_BUCKET_NAME",
        "OLLAMA_HOST",
        "PHONE_DEFAULT_COUNTRY_CODE",
        mode="before",
    )
    @classmethod
    def empty_string_to_none(cls, value: str | None):
        if value == "":
            return None
        return value

settings = Settings()
