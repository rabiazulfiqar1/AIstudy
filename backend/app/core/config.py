from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Config(BaseSettings):
    
    # Database (SQLAlchemy)
    DATABASE_URL: Optional[str] = None
    DB_FORCE_ROLL_BACK: bool = False
    
    # Supabase (for Auth/Storage)
    NEXT_PUBLIC_SUPABASE_URL: Optional[str] = None
    NEXT_PUBLIC_SUPABASE_ANON_KEY: Optional[str] = None
    SUPABASE_SERVICE_ROLE_KEY: Optional[str] = None
    
    # API Keys
    HF_TOKEN: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None
    GITHUB_TOKEN: Optional[str] = None
    
    # Other Services
    MONGO_URI: Optional[str] = None
    REDIS_URL: Optional[str] = None
    
    # Models
    HF_SUMMARIZATION_MODEL: str = "facebook/bart-large-cnn"
    HF_NOTE_GENERATION_MODEL: str = "google/flan-t5-base"  
    
    # Kaggle
    KAGGLE_USERNAME: Optional[str] = None
    KAGGLE_KEY: Optional[str] = None  

    # This is the modern syntax for Pydantic V2
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding='utf-8')

# Create a singleton config instance
config = Config()