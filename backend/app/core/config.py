from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    GEMINI_API_KEY: Optional[str] = None
    MONGO_URI: Optional[str] = None
    REDIS_URL: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    SUPABASE_URL: Optional[str] = None
    SUPABASE_DB_URL: Optional[str] = None
    SUPABASE_SERVICE_ROLE_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None
    HF_TOKEN: str | None = None 
    NEXT_PUBLIC_SUPABASE_URL: Optional[str] = None
    NEXT_PUBLIC_SUPABASE_ANON_KEY: Optional[str] = None

    # This is the modern syntax for Pydantic V2
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding='utf-8')

settings = Settings()