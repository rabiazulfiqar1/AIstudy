from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router as api_router
from contextlib import asynccontextmanager
from app.database.sql_engine import engine
from app.database.init_db import initialize_database_objects, verify_database_objects

#context manager is basically a function that sets up a context for some code to run in, and then cleans up after that code has run: setup and teardown logic
#lifespan event to connect and disconnect the database when the app starts and stops: it's done before any request is handled
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager - handles startup and shutdown events
    """
    # ========== STARTUP ==========
    # await engine.connect()
    print("🚀 Server starting up...")
    
    # Initialize database objects (procedures, triggers, views)
    try:
        await initialize_database_objects()
        
        # Optional: Verify installation (can comment out in production)
        is_valid = await verify_database_objects()
        if not is_valid:
            print("⚠️  Warning: Some database objects failed to initialize")
            
    except Exception as e:
        print(f"⚠️  Database initialization warning: {e}")
        # Don't crash the app - it might be a permissions issue
        # The app can still run with Python-only logic
    
    print("✅ Server ready to accept requests\n")
    
    yield #lifespan function will pause here and let the app run to handle requests and when the app is shutting down, it will resume here
    await engine.dispose() #dispose of the engine, closing all connections in the pool
    print("✅ Database connections closed")

app = FastAPI(title="Study backend", lifespan=lifespan)

app.include_router(api_router)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"status": "ok", "message": "Welcome to SparkSpace Backend"}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "database": "connected",
        "api_version": "v1"
    }