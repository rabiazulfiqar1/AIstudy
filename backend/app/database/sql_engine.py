from app.core.config import config
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from app.database.tables import metadata #noqa
from contextlib import asynccontextmanager

engine = create_async_engine(
    config.DATABASE_URL,
    pool_size=10,          # Reduce from default (usually 5)
    max_overflow=0,       # number of extra connections allowed beyond pool_size. These are created on demand and discarded after use.
    pool_pre_ping=True,   # checks if connections are alive before using them
    pool_recycle=3600,    # Recycle connections after 1 hour
    echo=False            # Disable SQL logging for performance
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    expire_on_commit=False,
    class_=AsyncSession,
    autocommit=False,
    autoflush=False
)

# Helper to create tables if not exist
async def create_tables():
    async with engine.begin() as conn:
        await conn.run_sync(metadata.create_all)

# Context manager for service functions
@asynccontextmanager
async def get_db():
    """
    Use this in service functions
    
    Example:
        async with get_db() as db:
            result = await db.execute(query)
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise

# Dependency for FastAPI routes
async def get_db_session():
    """
    Use this with Depends() in FastAPI routes
    
    Example:
        @router.get("/users")
        async def get_users(db: AsyncSession = Depends(get_db_session)):
            ...
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()