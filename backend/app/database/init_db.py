"""
Database Initialization Script
File: app/database/init_db.py

Loads and executes SQL procedures, triggers, and views on startup.
"""

from sqlalchemy import text
from pathlib import Path
from app.database.sql_engine import engine


async def initialize_database_objects():
    """
    Initialize database procedures, triggers, and views.
    
    This should be called once during application startup.
    It's idempotent - can be run multiple times safely (uses CREATE OR REPLACE).
    """
    
    database_dir = Path(__file__).parent  # app/database/
    
    sql_files = [
        ('triggers.sql', 'Database Triggers'),
        ('views.sql', 'Database Views')
    ]
    
    async with engine.begin() as conn:
        for filename, description in sql_files:
            sql_path = database_dir / filename
            
            if not sql_path.exists():
                print(f"⚠️  Warning: {filename} not found at {sql_path}")
                continue
            
            try:
                with open(sql_path, 'r') as f:
                    sql_content = f.read()
                    await conn.execute(text(sql_content))
                    print(f"✅ Loaded {description} from {filename}")
                    
            except Exception as e:
                print(f"❌ Error loading {filename}: {e}")
                # Don't raise - allow app to continue even if SQL objects fail
    
    print("✅ Database initialization complete")


async def verify_database_objects():
    """
    Verify that all database objects were created successfully.
    Useful for debugging deployment issues.
    """
    
    checks = []
    
    async with engine.connect() as conn:   
        # Check if views exist
        view_names = ['popular_projects_by_difficulty', 'top_skill_combinations']
        for view_name in view_names:
            result = await conn.execute(text("""
                SELECT EXISTS (
                    SELECT 1 FROM pg_views 
                    WHERE viewname = :view_name
                )
            """), {'view_name': view_name})
            checks.append((f'{view_name} view', result.scalar()))
        
        # Check if triggers exist
        result = await conn.execute(text("""
            SELECT COUNT(*) FROM pg_trigger 
            WHERE tgname LIKE 'trigger_%'
        """))
        trigger_count = result.scalar()
        checks.append(('Triggers', trigger_count > 0))
    
    print("\n📊 Database Objects Verification:")
    print("-" * 50)
    for name, exists in checks:
        status = "✅" if exists else "❌"
        print(f"{status} {name}: {'EXISTS' if exists else 'MISSING'}")
    print("-" * 50)
    
    return all(exists for _, exists in checks)


# For standalone testing
if __name__ == "__main__":
    import asyncio
    
    async def main():
        print("Initializing database objects...")
        await initialize_database_objects()
        print("\nVerifying installation...")
        success = await verify_database_objects()
        
        if success:
            print("\n✅ All database objects successfully installed!")
        else:
            print("\n⚠️  Some database objects are missing!")
    
    asyncio.run(main())