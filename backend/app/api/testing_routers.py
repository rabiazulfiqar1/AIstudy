from typing import Optional
from fastapi import Depends, HTTPException, Query, APIRouter
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete, insert
from datetime import datetime
from app.database.sql_engine import get_db_session
from app.database.tables import (
    projects,
    project_embeddings
)
from app.api.rs import ( 
    calculate_hybrid_score,
    calculate_skill_match_score,
    get_user_profile_data, 
    build_user_query_text,
    ensure_project_embedding,
    get_all_projects_with_skills,
    InteractionCreate,  
    user_project_interactions,
    project_skills,
    skills,
    find_similar_projects_vector
)

router = APIRouter()

# ------ RECOMMENDATIONS (VECTOR-ENHANCED) ------

@router.get("/recommendations/{user_id}")
async def get_recommendations(
    user_id: str,
    limit: int = Query(10, le=50),
    algorithm: str = Query("hybrid", regex="^(hybrid|semantic|traditional)$"),
    source_filter: Optional[str] = Query(None, regex="^(github|kaggle_competition|kaggle_dataset|curated|all)$"),
    db: AsyncSession = Depends(get_db_session)
):
    """
    Get personalized project recommendations
    
    Algorithms:
    - hybrid (default): Combines semantic search + skill matching
    - semantic: Pure vector similarity
    - traditional: Legacy keyword/skill matching
    
    Source Filters:
    - github: GitHub repositories
    - kaggle_competition: Kaggle competitions
    - kaggle_dataset: Kaggle datasets (analysis projects)
    - curated: Hand-picked projects
    - all (default): All sources
    """
    
    # Get user profile
    user_profile = await get_user_profile_data(user_id, db)
    
    if not user_profile:
        raise HTTPException(status_code=404, detail="User profile not found")
    
    # Parse source filter
    sources = None
    if source_filter and source_filter != 'all':
        sources = [source_filter]
    
    recommendations = []
    
    if algorithm in ["hybrid", "semantic"]:
        # Build semantic query from user profile
        user_query = build_user_query_text(user_profile)
        
        # Get semantically similar projects
        similar_projects = await find_similar_projects_vector(
            user_query=user_query,
            db=db,
            limit=limit * 2, # Get more for filtering
            source_filter=sources
        )
        
        # Enrich with skills
        for project in similar_projects[:limit]:
            # Get project skills
            result = await db.execute(
                select(project_skills, skills.c.name)
                .join(skills, project_skills.c.skill_id == skills.c.id)
                .where(project_skills.c.project_id == project['id'])
            )
            
            project['project_skills'] = [
                {'name': row.name, 'is_required': row.is_required}
                for row in result.fetchall()
            ]
            
            if algorithm == "hybrid":
                # Calculate hybrid score
                score, matching, missing, reason = calculate_hybrid_score(
                    user_profile,
                    project,
                    project['semantic_similarity']
                )
            else:
                # Pure semantic
                score = project['semantic_similarity']
                matching, missing = [], []
                reason = "Semantically similar to your profile"
            
            recommendations.append({
                'project_id': project['id'],
                'title': project['title'],
                'description': project['description'],
                'repo_url': project.get('repo_url'),
                'difficulty': project['difficulty'],
                'topics': project['topics'],
                'estimated_hours': project['estimated_hours'],
                'match_score': round(score * 100, 1),
                'matching_skills': matching,
                'missing_skills': missing,
                'reason': reason,
                'semantic_similarity': round(project['semantic_similarity'] * 100, 1)
            })
        
        # Re-sort by hybrid score if using hybrid
        if algorithm == "hybrid":
            recommendations.sort(key=lambda x: x['match_score'], reverse=True)
    
    else:
        # Traditional algorithm (legacy)
        all_projects = await get_all_projects_with_skills(db)
        
        if sources:
            all_projects = [p for p in all_projects if p.get('source') in sources]
        
        scored_projects = []
        for project in all_projects:
            skill_score, matching, missing = calculate_skill_match_score(user_profile, project)
            
            scored_projects.append({
                'project': project,
                'score': skill_score,
                'matching_skills': matching,
                'missing_skills': missing
            })
        
        scored_projects.sort(key=lambda x: x['score'], reverse=True)
        
        for item in scored_projects[:limit]:
            p = item['project']
            recommendations.append({
                'project_id': p['id'],
                'title': p['title'],
                'description': p['description'],
                'repo_url': p.get('repo_url'),
                'difficulty': p['difficulty'],
                'topics': p['topics'],
                'estimated_hours': p['estimated_hours'],
                'match_score': round(item['score'] * 100, 1),
                'matching_skills': item['matching_skills'],
                'missing_skills': item['missing_skills'],
                'reason': "Skill-based match"
            })
    
    return {
        "recommendations": recommendations,
        "algorithm": algorithm,
        "source_filter": source_filter or "all",
        "user_profile_summary": {
            "skill_level": user_profile.get('skill_level'),
            "skills_count": len(user_profile['skills']),
            "interests": user_profile.get('interests', [])
        }
    }

# ------ INTERACTIONS ------

@router.post("/interactions/{user_id}")
async def log_interaction(
    user_id: str,
    interaction: InteractionCreate,
    db: AsyncSession = Depends(get_db_session)
):
    """Log user interaction with project"""
    
    await db.execute(
        insert(user_project_interactions).values(
            user_id=user_id,
            project_id=interaction.project_id,
            interaction_type=interaction.interaction_type,
            rating=interaction.rating,
            created_at=datetime.utcnow()
        )
    )
    
    return {"message": "Interaction logged successfully"}

@router.get("/interactions/{user_id}")
async def get_user_interactions(
    user_id: str,
    db: AsyncSession = Depends(get_db_session)
):
    """Get user's project interactions"""
    
    result = await db.execute(
        select(user_project_interactions, projects.c.title)
        .join(projects, user_project_interactions.c.project_id == projects.c.id)
        .where(user_project_interactions.c.user_id == user_id)
        .order_by(user_project_interactions.c.created_at.desc())
    )
    
    interactions = [
        {
            'id': row.id,
            'project_id': row.project_id,
            'project_title': row.title,
            'interaction_type': row.interaction_type,
            'rating': row.rating,
            'created_at': row.created_at
        }
        for row in result.fetchall()
    ]
    
    return {"interactions": interactions}

# ------ EMBEDDING MANAGEMENT ------

@router.post("/admin/embeddings/generate")
async def generate_all_embeddings(
    db: AsyncSession = Depends(get_db_session)
):
    """Generate embeddings for all projects (admin endpoint)"""
    
    result = await db.execute(select(projects.c.id))
    project_ids = [row[0] for row in result.fetchall()]
    
    generated = 0
    skipped = 0
    
    for project_id in project_ids:
        result = await db.execute(
            select(project_embeddings).where(project_embeddings.c.project_id == project_id)
        )
        
        if result.first():
            skipped += 1
            continue
        
        await ensure_project_embedding(project_id, db)
        generated += 1
    
    return {
        "message": "Embedding generation complete",
        "generated": generated,
        "skipped": skipped,
        "total": len(project_ids)
    }

@router.post("/admin/embeddings/regenerate/{project_id}")
async def regenerate_project_embedding(
    project_id: int,
    db: AsyncSession = Depends(get_db_session)
):
    """Regenerate embedding for a specific project"""
    
    # Delete existing
    await db.execute(
        delete(project_embeddings).where(project_embeddings.c.project_id == project_id)
    )
    
    # Generate new
    await ensure_project_embedding(project_id, db)
    
    return {"message": f"Embedding regenerated for project {project_id}"}


# async def find_similar_projects_vector(
#     user_query: str,
#     db: AsyncSession,
#     limit: int = 20,
#     difficulty_filter: Optional[str] = None,
#     source_filter: Optional[List[str]] = None  
# ) -> List[dict]:
#     """
#     Find similar projects using pgvector cosine similarity
    
#     Args:
#         user_query: Text query for semantic search
#         db: Database session
#         limit: Maximum results to return
#         difficulty_filter: Filter by difficulty level
#         source_filter: Filter by source (e.g., ['github', 'kaggle_competition'])
    
#     Returns projects sorted by semantic similarity
#     """
#     # Generate query embedding
#     query_embedding = embedding_service.encode(user_query)
    
#     # Convert to list
#     embedding_list = query_embedding.tolist() if hasattr(query_embedding, 'tolist') else list(query_embedding)
    
#     # Cast embedding to vector type
#     embedding_vector = cast(embedding_list, Vector)
    
#     # Build query using SQLAlchemy
#     distance = project_embeddings.c.embedding.cosine_distance(embedding_vector).label('distance')
#     similarity = (1 - project_embeddings.c.embedding.cosine_distance(embedding_vector)).label('similarity')
    
#     stmt = (
#         select(
#             projects,
#             project_embeddings.c.embedding, 
#             distance,
#             similarity
#         )
#         .select_from(projects)
#         .join(project_embeddings, projects.c.id == project_embeddings.c.project_id)
#     )
    
#     # Add difficulty filter if specified
#     if difficulty_filter:
#         stmt = stmt.where(projects.c.difficulty == difficulty_filter)
    
#     if source_filter:
#         stmt = stmt.where(projects.c.source.in_(source_filter))
    
#     # Order by distance and limit
#     stmt = stmt.order_by(distance).limit(limit)
    
#     result = await db.execute(stmt)
    
#     projects_with_similarity = []
#     for row in result:
#         project_data = {
#             'id': row.id,
#             'title': row.title,
#             'description': row.description,
#             'repo_url': row.repo_url,
#             'difficulty': row.difficulty,
#             'topics': row.topics,
#             'estimated_hours': row.estimated_hours,
#             'source': row.source,
#             'stars': row.stars,
#             'language': row.language,
#             'semantic_similarity': float(row.similarity)
#         }
#         projects_with_similarity.append(project_data)
    
#     return projects_with_similarity