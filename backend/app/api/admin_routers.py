from fastapi import Depends, APIRouter, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text, func
from app.database.sql_engine import get_db_session
from app.database.tables import (
    projects,
    project_embeddings
)
from app.api.rs import (
    user_profiles,
    popular_projects_view,
    top_skill_combinations_view
)

router = APIRouter()

@router.get("/stats/projects-by-difficulty")
async def get_project_stats(db: AsyncSession = Depends(get_db_session)):
    """Get project statistics from the view"""
    
    from sqlalchemy import select
    
    result = await db.execute(
        select(popular_projects_view)
    )
    
    stats = [
        {
            "difficulty": row.difficulty,
            "project_count": row.project_count,
            "avg_stars": float(row.avg_stars or 0),
            "avg_hours": float(row.avg_hours or 0)
        }
        for row in result.fetchall()
    ]
    
    return {"difficulty_stats": stats}


@router.get("/stats/skill-combinations")
async def get_popular_skill_combos(
    limit: int = 10,
    db: AsyncSession = Depends(get_db_session)
):
    """Get most common skill combinations from view"""
    
    from sqlalchemy import select
    
    result = await db.execute(
        select(top_skill_combinations_view).limit(limit)
    )
    
    combos = [
        {
            "skills": [row.skill_1_name, row.skill_2_name],
            "project_count": row.project_count
        }
        for row in result.fetchall()
    ]
    
    return {"top_combinations": combos}
    
# Check activity log (populated by triggers)
@router.get("/users/{user_id}/activity")
async def get_user_activity(
    user_id: str,
    db: AsyncSession = Depends(get_db_session)
):
    """Get user activity log (populated by profile update trigger)"""
    
    query = text("""
        SELECT activity_type, details, created_at
        FROM user_activity_log
        WHERE user_id = :user_id
        ORDER BY created_at DESC
        LIMIT 50
    """)
    
    result = await db.execute(query, {"user_id": user_id})
    
    activities = [
        {
            "type": row.activity_type,
            "details": row.details,
            "timestamp": row.created_at
        }
        for row in result.fetchall()
    ]
    
    return {"activities": activities}

@router.get("/stats")
async def get_system_stats(db: AsyncSession = Depends(get_db_session)):
    """Get system statistics"""
    
    # Project counts
    result = await db.execute(select(func.count()).select_from(projects))
    total_projects = result.scalar()
    
    # Embeddings count
    result = await db.execute(select(func.count()).select_from(project_embeddings))
    total_embeddings = result.scalar()
    
    # By difficulty
    result = await db.execute(
        select(projects.c.difficulty, func.count())
        .group_by(projects.c.difficulty)
    )
    by_difficulty = {row[0]: row[1] for row in result}
    # by source
    result = await db.execute(
        select(projects.c.source, func.count())
        .group_by(projects.c.source)
    )
    by_source = {row[0]: row[1] for row in result}
    
    # User profiles count
    result = await db.execute(select(func.count()).select_from(user_profiles))
    total_users = result.scalar()
    
    return {
        "projects": {
            "total": total_projects,
            "with_embeddings": total_embeddings,
            "embedding_coverage": round(total_embeddings / total_projects * 100, 1) if total_projects > 0 else 0,
            "by_difficulty": by_difficulty,
            "by_source": by_source 
        },
        "users": {
            "total_profiles": total_users
        },
        "embedding_model": "all-MiniLM-L6-v2"
    }

@router.post("/admin/cleanup-interactions")
async def cleanup_old_interactions(
    days_threshold: int = Query(180, ge=30, le=730),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Clean up old 'viewed' interactions to maintain performance.
    
    Admin endpoint to remove old view records while keeping
    important interactions (bookmarks, started, completed).
    
    Requires admin authentication in production!
    """
    
    result = await db.execute(
        text("""
            SELECT clean_old_interactions(:days)
        """),
        {"days": days_threshold}
    )
    
    deleted_count = result.scalar()
    
    await db.commit()
    
    return {
        "message": "Cleanup completed",
        "deleted_interactions": deleted_count,
        "days_threshold": days_threshold
    }


@router.get("/admin/projects-needing-embeddings")
async def get_projects_needing_embeddings(
    db: AsyncSession = Depends(get_db_session)
):
    """
    Get list of projects that don't have embeddings yet.
    
    Useful for batch embedding generation.
    """
    
    result = await db.execute(
        text("""
            SELECT * FROM get_projects_needing_embeddings()
        """)
    )
    
    projects = []
    for row in result:
        projects.append({
            "project_id": row.project_id,
            "embedding_text": row.embedding_text
        })
    
    return {
        "projects_without_embeddings": projects,
        "count": len(projects)
    }