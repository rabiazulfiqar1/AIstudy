from pydantic import BaseModel
from typing import Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, insert
from datetime import datetime
from app.database.sql_engine import get_db_session
from app.database.tables import users

router = APIRouter()

class BasicProfileCreate(BaseModel):
    full_name: str  # Required now
    username: str   # Required now
    organization: Optional[str] = None
    field_of_study: Optional[str] = None
    phone: Optional[str] = None
    profile_pic: Optional[str] = None

@router.get("/users/{user_id}/basic-profile")
async def get_basic_profile(
    user_id: str,
    db: AsyncSession = Depends(get_db_session)
):
    """Get basic user profile info"""
    try:
        user_uuid = UUID(user_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid user ID")
    
    result = await db.execute(
        select(users).where(users.c.user_id == user_uuid)
    )
    user_row = result.first()
    
    if not user_row:
        raise HTTPException(status_code=404, detail="User not found")
    
    return dict(user_row._mapping)

@router.post("/users/{user_id}/basic-profile")
async def update_basic_profile(
    user_id: str,
    profile: BasicProfileCreate,
    db: AsyncSession = Depends(get_db_session)
):
    """Create or update basic user profile info"""
    try:
        user_uuid = UUID(user_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid user ID")
    
    try:
        # Check if user exists
        result = await db.execute(
            select(users).where(users.c.user_id == user_uuid)
        )
        existing_user = result.first()
        
        # Get email from Supabase auth (you'll need to pass this from frontend)
        # For now, we'll require email in the request or handle it separately
        
        if existing_user:
            # Update existing user
            await db.execute(
                update(users)
                .where(users.c.user_id == user_uuid)
                .values(
                    full_name=profile.full_name,
                    username=profile.username,
                    organization=profile.organization,
                    field_of_study=profile.field_of_study,
                    phone=profile.phone,
                    profile_pic=profile.profile_pic,
                    updated_at=datetime.utcnow()
                )
            )
        else:
            # Create new user - This needs email!
            raise HTTPException(
                status_code=404, 
                detail="User not found. Please use the create user endpoint first."
            )
        
        await db.commit()
        return {"message": "Profile updated successfully"}
        
    except Exception as e:
        await db.rollback()
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Add endpoint to create user initially
@router.post("/users/create")
async def create_user(
    user_data: dict,
    db: AsyncSession = Depends(get_db_session)
):
    """Create a new user record (call this after Supabase signup)"""
    try:
        user_id = UUID(user_data['user_id'])
        email = user_data['email']
        username = user_data.get('username', email.split('@')[0])  # Default username from email
        full_name = user_data.get('full_name', '')
        
        # Check if user already exists
        result = await db.execute(
            select(users).where(users.c.user_id == user_id)
        )
        
        if result.first():
            return {"message": "User already exists"}
        
        # Insert new user
        await db.execute(
            insert(users).values(
                user_id=user_id,
                email=email,
                username=username,
                full_name=full_name,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
        )
        await db.commit()
        
        return {"message": "User created successfully"}
        
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=str(e))