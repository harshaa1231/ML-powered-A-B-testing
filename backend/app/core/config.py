from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # App
    environment: str = "development"
    app_name: str = "AB Testing Pro API"

    # Database (async driver for the app, sync driver for Alembic)
    database_url: str = "postgresql+asyncpg://abtesting:abtesting@localhost:5432/abtesting"
    database_url_sync: str = "postgresql+psycopg2://abtesting:abtesting@localhost:5432/abtesting"

    # Auth
    jwt_secret: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24 * 7  # 7 days

    # CORS
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]

    # RAG / GenAI
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    rag_top_k: int = 4

    # Storage
    storage_dir: str = "./storage"


@lru_cache
def get_settings() -> Settings:
    return Settings()
