"""Cache operations using SQLAlchemy"""
import json
from typing import Optional, List
from sqlalchemy import text, delete
from app.database.sql_engine import get_db
from app.database.tables import video_cache as VideoCache

# :cache_key is a placeholder — SQLAlchemy will replace it safely with the Python variable.
# RETURNING lets you get back rows that were inserted/updated/deleted.
# ON CONFLICT ... DO UPDATE Insert a row. If it already exists (unique constraint), update it instead.
# .rowcount is SQLAlchemy’s way to get how many rows were affected, avoids fetching all rows: equivalent to SELECT COUNT(*)
# INTERVAL '1 day' in Postgres means a time duration of 1 day
# EXCLUDED represents the values we were trying to insert allowing us to reference them in the update part of the upsert.

async def save_to_cache(cache_key: str, cache_type: str, data: dict) -> bool:
    """Save data to cache"""
    try:
        async with get_db() as db:
            query = text("""
                INSERT INTO video_cache (id, cache_key, cache_type, data, created_at, updated_at, last_accessed)
                VALUES (gen_random_uuid(), :cache_key, :cache_type, :data, NOW(), NOW(), NOW())
                ON CONFLICT (cache_key, cache_type) 
                DO UPDATE SET 
                    data = EXCLUDED.data,
                    updated_at = NOW(),
                    last_accessed = NOW()
            """)
            
            await db.execute(query, {
                "cache_key": cache_key,
                "cache_type": cache_type,
                "data": json.dumps(data)
            })
            
            print(f"✅ Cached: {cache_key}_{cache_type}")
            return True
    except Exception as e:
        print(f"❌ Cache save failed: {e}")
        return False

async def load_from_cache(cache_key: str, cache_type: str) -> Optional[dict]:
    """Load data from cache"""
    try:
        async with get_db() as db:
            query = text("""
                UPDATE video_cache 
                SET last_accessed = NOW()
                WHERE cache_key = :cache_key AND cache_type = :cache_type
                RETURNING data
            """)
            
            result = await db.execute(query, {
                "cache_key": cache_key,
                "cache_type": cache_type
            })
            row = result.fetchone()
            
            if row:
                print(f"✅ Cache HIT: {cache_key}_{cache_type}")
                return row[0]  # JSONB automatically parsed
            
            print(f"❌ Cache MISS: {cache_key}_{cache_type}")
            return None
    except Exception as e:
        print(f"❌ Cache load failed: {e}")
        return None

async def clear_old_cache(max_age_days: int = 7) -> int:
    """Delete old cache entries"""
    try:
        async with get_db() as db:
            query = text("""
                DELETE FROM video_cache 
                WHERE updated_at < NOW() - INTERVAL '1 day' * :days
            """)
            
            result = await db.execute(query, {"days": max_age_days})
            deleted_count = result.rowcount or 0
            
            if deleted_count > 0:
                print(f"🗑️ Deleted {deleted_count} old cache entries")
            
            return deleted_count
    except Exception as e:
        print(f"❌ Failed to clear old cache: {e}")
        return 0


async def clear_cache_by_key(cache_key: str) -> int:
    """Delete all entries for a cache key"""
    try:
        async with get_db() as db:
            stmt = delete(VideoCache).where(VideoCache.cache_key == cache_key)
            result = await db.execute(stmt)
            return result.rowcount
    except Exception as e:
        print(f"❌ Failed to clear cache by key: {e}")
        return 0

async def get_all_cache_keys(limit: int = 100) -> List[str]:
    """Get list of cache keys"""
    try:
        async with get_db() as db:
            query = text("""
                SELECT DISTINCT cache_key 
                FROM video_cache 
                ORDER BY cache_key
                LIMIT :limit
            """)
            
            result = await db.execute(query, {"limit": limit})
            return [row[0] for row in result.fetchall()]
    except Exception as e:
        print(f"❌ Failed to get cache keys: {e}")
        return []
    
    
# # Local file-based cache (deprecated) - kept for reference

# def save_to_cache_local(cache_key: str, cache_type: str, data: dict):
#     """Save processed data to cache"""
#     cache_path = get_cache_path(cache_key, cache_type)
#     try:
#         with open(cache_path, 'wb') as f:
#             pickle.dump(data, f)
#         print(f"✅ Cached to: {cache_path}")
#     except Exception as e:
#         print(f"❌ Cache save failed: {e}")

# def load_from_cache_local(cache_key: str, cache_type: str) -> Optional[dict]:
#     """Load processed data from cache"""
#     cache_path = get_cache_path(cache_key, cache_type)
    
#     if cache_path.exists():
#         try:
#             with open(cache_path, 'rb') as f:
#                 data = pickle.load(f)
#             print(f"✅ Cache HIT: {cache_path}")
#             return data
#         except Exception as e:
#             print(f"❌ Cache load failed: {e}")
#             return None
    
#     print(f"❌ Cache MISS: {cache_path}")
#     return None

# def clear_old_cache_local(max_age_days: int = 7):
#     """Delete cache files older than max_age_days"""
#     import time
#     current_time = time.time()
#     max_age_seconds = max_age_days * 86400 #24(hrs)*60(mins)*60(secs) = 86400 seconds in a day
    
#     deleted_count = 0
#     for cache_file in CACHE_DIR.glob("*.pkl"):
#         if current_time - cache_file.stat().st_mtime > max_age_seconds:
#             cache_file.unlink()
#             deleted_count += 1
    
#     if deleted_count > 0:
#         print(f"🗑️ Deleted {deleted_count} old cache files")