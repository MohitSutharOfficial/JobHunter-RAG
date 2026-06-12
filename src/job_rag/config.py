"""Application configuration via environment variables / .env file."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM: Groq is preferred when set; Gemini is used as fallback LLM.
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    # Gemini: used for embeddings, and as LLM when no Groq key is set.
    google_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    embedding_model: str = "models/text-embedding-004"

    chroma_dir: str = "./data/chroma"
    collection_name: str = "job_postings"
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 5
    use_fake_embeddings: bool = False
    resume_path: str = "./data/resume.txt"


@lru_cache
def get_settings() -> Settings:
    return Settings()
