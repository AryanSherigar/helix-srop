import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "development"
    log_level: str = "INFO"
    secret_key: str = "change-me-in-prod"

    database_url: str = "sqlite+aiosqlite:///./helix_srop.db"
    chroma_persist_dir: str = "./chroma_db"

    google_api_key: str = ""
    adk_model: str = "gemini-3.1-flash-lite-preview"

    llm_timeout_seconds: int = 120
    tool_timeout_seconds: int = 60

    rerank_enabled: bool = True
    rerank_candidate_k: int = 20


settings = Settings()

if settings.google_api_key:
    os.environ.setdefault("GOOGLE_API_KEY", settings.google_api_key)
