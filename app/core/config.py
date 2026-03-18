import os
from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "AI CSV Cleaning Agent"
    VERSION: str = "1.0.0"
    
    # Storage settings
    STORAGE_DIR: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "storage")
    UPLOAD_DIR: str = os.path.join(STORAGE_DIR, "uploads")
    OUTPUT_DIR: str = os.path.join(STORAGE_DIR, "outputs")
    
    # LLM Settings
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3" # Default model, user can change
    
    # Database settings
    DATABASE_URL: str = "postgresql+asyncpg://postgres:admin@localhost:5432/ai_cleaning_db"
    
    # Processing settings
    CHUNK_SIZE: int = 1000 # For processing large CSVs chunk by chunk
    AI_BATCH_SIZE: int = 50
    CLEAN_PREVIEW_ROWS: int = 25
    CONFIDENCE_AUTO_ACCEPT: float = 0.95
    CONFIDENCE_REVIEW: float = 0.75
    DATE_OUTPUT_FORMAT: str = "%Y-%m-%d"
    DATETIME_OUTPUT_FORMAT: str = "%Y-%m-%dT%H:%M:%S"
    PHONE_DEFAULT_COUNTRY_CODE: Optional[str] = None
    NORMALIZE_PERCENTAGE_TO_FRACTION: bool = False
    NULL_TOKENS: str = "n/a,na,null,none,-,--,,n.a.,nil,missing,not available,not provided,blank,#n/a,nan,tbd"
    NULL_OUTPUT_TOKEN: str = "N/A"
    
    class Config:
        env_file = ".env"

settings = Settings()

# Ensure storage directories exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
