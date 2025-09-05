from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    OPENAI_API_KEY: str = ""
    SUPABASE_URL: str = ""
    SUPABASE_DB_URL: str = ""
    SUPABASE_SERVICE_ROLE_KEY: str = ""
    GEMINI_API_KEY: str  # ✨ ADDED this line
    GROQ_API_KEY: str = ""

    
    # These two lines are now active and will fix your error
    MONGO_URI: str
    REDIS_URL: str

    # This is the modern syntax for Pydantic V2
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding='utf-8')

settings = Settings()