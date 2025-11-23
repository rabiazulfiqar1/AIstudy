import asyncio
from app.database.sql_engine import create_tables

async def main():
    print("🔄 Creating tables...")
    await create_tables()
    print("✅ Done!")

if __name__ == "__main__":
    asyncio.run(main())