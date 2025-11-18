from supabase import create_client
from app.core.config import config

supabase = create_client(config.NEXT_PUBLIC_SUPABASE_URL, config.NEXT_PUBLIC_SUPABASE_ANON_KEY)